"""Contracts for the onion's mirror: the turns it holds are the conversation's own, by identity, never by count.

The librarian mirrors a saved conversation into a Flesh, and the queue moves
its oldest spans into the Cellar under receipts. A retry deletes the last
answer and its question and writes both again; a synchronisation rewrites
the whole conversation from another device, every turn legacy; a message
can be deleted. Each leaves the conversation with as many turns as before,
or fewer, and other words in them. These contracts hold that the mirror
follows the conversation by what each turn is -- its role, its words, its
origin and its segments -- and that it never rewrites what the Cellar
keeps:

  * MI1 -- after a retry, the Flesh is the conversation the store holds: the
    rejected answer is gone and the answer given instead is in.
  * MI2 -- after a synchronisation, the Flesh is the conversation the store
    holds; the same synchronisation applied again changes nothing.
  * MI3 -- a divergence inside a span already in the Cellar supersedes that
    span's receipt and every later one, keeps every earlier one, and leaves
    every span of the Cellar as it was, byte for byte.
  * MI4 -- after it, what the state mirrors is the conversation again: the
    spans still standing, then the conversation from the first turn of the
    first span superseded.
  * MI5 -- a superseded span's peel leaves the memory block, and the span
    stays recallable by the user.
  * MI6 -- a turn id is never given twice, across a save and a load.
  * MI7 -- a step whose span was taken back while its summary was written
    evicts nothing, though the turns written again take the same places.
  * MI8 -- a proposal whose span is superseded is superseded with it:
    neither offered nor taken.
  * MI9 -- a conversation that lost its last turns takes them back from the
    Flesh.
  * MI10 -- an empty conversation changes nothing: forgetting is another
    verb.
  * MI11 -- the mirror counts the turns it appends, the turns it takes back
    and the receipts it supersedes; each leaves zero.
  * MI12 -- what the mirror logs of a divergence names turns and counts,
    never a word of a turn.
  * MI13 -- after a step, a state holds at most the refusal mark of the span
    at the head of its Flesh: a span that leaves takes its mark with it.
  * MI14 -- a mark kept with its span at the head spares the second asking.
  * MI15 -- through random histories of appends, retries, deletions,
    synchronisations and edits, with queue steps in between, the mirror
    stays the conversation, the Cellar never changes, no turn id is given
    twice and no superseded receipt reaches the block; the same histories
    find the defect of a mirror that counts.
  * MI16 -- a superseded receipt, a superseded proposal and the count of
    turn ids given come back from the store as saved, under the same root.
  * MI17 -- once the Flesh is taken back, a burst fires after the configured
    growth from what is left, not from the count before.
  * MI18 -- LB7 word for word, but for its last clause: the mirror is exact
    and idempotent on appends, and a shorter history with malformed
    messages in it takes back what it no longer holds instead of being
    ignored (LB7 is deselected by name; its last clause held the blind
    mirror the review asked to remove).
  * MI19 -- a proposal left open on a superseded span, as a file written by
    another hand can hold it, is neither offered nor taken.
  * MI20 -- the queue makes no proposal from a receipt superseded between
    its step and its offer.
  * MI21 -- a superseded receipt leaves the digest the window shows and the
    user's list of open receipts.
  * MI22 -- the mirror reads what the state holds, never what it remembers
    of it: a turn another writer took back is mirrored again.
  * MI23 -- the anchors of a held span that is superseded leave the block:
    the model is no longer shown words the conversation no longer holds.
  * MI24 -- a proposal is not made from a span the mirror supersedes while
    the proposal is being drawn: no proposal stays open on it, and the day's
    room is untouched.
  * MI25 -- a decision made again after its supersession is offered the
    same day: the same words, made again from the turns mirrored again,
    take the place of the offer the user already had, and no more room.
  * MI26 -- the user's verdict follows the decision, not its place: a
    decision declined or accepted is not offered again once its turn is
    mirrored again after a supersession, and the queue counts each one it
    holds back.
  * MI27 -- a Flesh row the mirror cannot read as a turn is taken back and
    mirrored again: the onion never stops following the conversation.
  * MI28 -- through random histories of typed decisions, regenerated and
    edited answers, queue steps, the user's verdicts and passing days, no
    proposal stays open on a superseded span, no day makes the user more
    offers than its cap -- an offer made again in the same words after its
    supersession counting once -- and no decision the user decided is first
    offered after the verdict; the same histories find a queue that forgets
    the verdicts, a mirror that leaves superseded proposals open, and a day
    with no cap.
  * MI29 -- a verdict holds back the copies of its decision not yet
    offered: the same turn sent twice makes two proposals before the user
    decides, the second deferred by the cap; declining the first, the
    second is never offered, that day or a later one.
  * MI30 -- a Flesh row whose segments are not a list -- null, zero, empty
    text, a mapping, false -- is taken back and mirrored again; a row
    written before turns had segments is kept as it stands.
  * MI31 -- edits never offer more in a day than its cap: a decision whose
    turn is edited again and again the same day, each time in other words,
    does not offer a new one each time; the last one waits for a later day.
  * MI32 -- an offer from an earlier day, made again in the same words after
    an edit, keeps its place and its day: it neither waits for a later day
    nor takes the room of the day it is made again on.
  * MI33 -- a listing on a full day draws the probes of what it shows, and
    of no deferred proposal it cannot open.
  * MI34 to MI37 -- supersede MI3, MI4, MI12 and MI21, whose base was written
    here: once every turn a peer's copy brings is received, a synchronisation
    changes every turn of a conversation written here, and the divergence
    those four aimed at moved to the first turn. Each keeps everything its
    predecessor held, over a conversation received from a peer before the
    edit, so that the edited turn alone diverges.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source, the conversation store over plain
sqlite in a temporary directory; the registry is blocked, so a summariser is
always injected.
"""

import hashlib
import itertools
import json
import logging
import random
import sqlite3
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_ONION = ("probes", "core_store", "receipts", "composer", "peels", "librarian")


def _plain_connect(path, **kw):
    return sqlite3.connect(path, check_same_thread=kw.get("check_same_thread", False))


def _window(tmp_path=None, persisted=False):
    """The onion's modules, and with ``tmp_path`` the conversation store over plain sqlite there."""
    targets = {f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in _ONION}
    if persisted:
        targets["opti_oignon.memory.onion_store"] = source("memory", "onion_store.py")
    seeded, blocked = {}, ("opti_oignon.inference_backend", "opti_oignon.db_utils")
    packages = ("opti_oignon.memory",)
    if tmp_path is not None:
        db = types.ModuleType("opti_oignon.db_utils")
        db.safe_connect = _plain_connect
        cfg = types.ModuleType("opti_oignon.config")
        cfg.DATA_DIR = Path(tmp_path)
        # The store publishes nothing here: no peer framework answers.
        guard = types.ModuleType("opti_oignon.veilid.guard")
        guard.veilid_available = lambda: False
        targets = {"opti_oignon.conversation": source("conversation.py"), **targets}
        seeded = {"opti_oignon.db_utils": db, "opti_oignon.config": cfg, "opti_oignon.veilid.guard": guard}
        blocked = ("opti_oignon.inference_backend",)
        packages += ("opti_oignon.veilid",)
    loaded, restore = isolate(targets=targets, blocked=blocked, seeded=seeded, packages=packages)
    lib = loaded["opti_oignon.memory.librarian"]
    lib.reset_librarian()
    conv = loaded.get("opti_oignon.conversation")
    if conv is not None:
        conv._encrypt = lambda v: "E:" + v
        conv._decrypt = lambda v: v[2:] if isinstance(v, str) and v.startswith("E:") else v
    return conv, lib, loaded, restore


def _conversation(conv, tmp_path):
    mgr = conv.ConversationManager(db_path=Path(tmp_path) / "c.db")
    return mgr, mgr.create_conversation(title="t").id


def _line(i):
    # Turns that decide nothing, each its own words: a summary that repeats
    # them asserts no decision, and no two spans share a peel's text.
    return f"Turn {i}: Alice reviewed service {i} on 2026-03-{i % 28 + 1:02d} and service {i} lives on cluster {i}."


def _say(mgr, cid, first, count, *, typed=False):
    """``count`` turns from number ``first``, user and assistant in turn; legacy unless ``typed``."""
    for i in range(first, first + count):
        role = "user" if i % 2 else "assistant"
        if typed:
            mgr.add_message(cid, role, _line(i), origin="typed" if role == "user" else "assistant")
        else:
            mgr.add_message(cid, role, _line(i))


def _base(mgr, cid):
    """The conversation as the store holds it, turn by turn: role, words, origin, segments."""
    return [(m["role"], m["content"], m["origin"], list(m["segments"] or [])) for m in mgr.get_mirror_messages(cid)]


def _shape(turns):
    return [(t["role"], t["text"], t["origin"], [list(s) for s in t.get("segments") or []]) for t in turns]


def _live(state):
    """What the state mirrors now: the spans of the receipts not superseded, in order, then the Flesh."""
    with state.lock:
        spans = [t for r in state.ledger.all() if r.kind != "superseded" for t in state.cellar.get(r.key)]
        return _shape(spans + state.flesh.turns())


def _payload(mgr, cid, *, edit=None, drop_last=0):
    """The conversation as a peer publishes it, with the edits ``edit`` names by index."""
    words = [(m["role"], m["content"]) for m in mgr.get_mirror_messages(cid)]
    for index, text in (edit or {}).items():
        words[index] = (words[index][0], text)
    if drop_last:
        words = words[:-drop_last]
    return {"conversation": {"id": cid, "title": "t", "messages": [{"role": r, "content": c} for r, c in words]}}


def _faithful(turns):
    return " ".join(t["text"] for t in turns)


def _gate(loaded, span_turns=2):
    return loaded["opti_oignon.memory.peels"].Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=span_turns)


def _budget(loaded, flesh=140):
    composer = loaded["opti_oignon.memory.composer"]
    return composer.Budget(window=780, reserve=60, core=60, receipts=300, peels=160, flesh=flesh, turn=60)


def _evict(lib, loaded, state, steps, summarize=_faithful):
    for _ in range(steps):
        outcome = lib.curate(state, summarize, gate=_gate(loaded), budget=_budget(loaded, flesh=1))
        assert outcome.evicted, f"control: the step evicts ({outcome.reason})"


def _cellar_bytes(state):
    with state.lock:
        return {key: json.dumps(state.cellar.get(key), sort_keys=True, ensure_ascii=False) for key in state.cellar.keys()}


def _ids(state):
    with state.lock:
        return [t["turn_id"] for t in state.flesh.turns()]


# ---------------------------------------------------------------------------
# MI1-MI2 -- a retry, a synchronisation: the Flesh is the conversation
# ---------------------------------------------------------------------------
def test_mi1_after_a_retry_the_flesh_is_the_conversation_the_store_holds(tmp_path):
    conv, lib, loaded, restore = _window(tmp_path)
    try:
        mgr, cid = _conversation(conv, tmp_path)
        _say(mgr, cid, 1, 4, typed=True)
        rejected, kept = "The answer the user rejected: service 4 moves to Lyon.", "The answer given instead."
        mgr.add_message(cid, "user", "And service 5?", origin="typed")
        mgr.add_message(cid, "assistant", rejected, origin="assistant")
        state = lib.state_for(cid)
        state.mirror(mgr.get_mirror_messages(cid))
        assert _shape(state.flesh.turns()) == _base(mgr, cid), "control: mirrored as saved"
        # The retry route: the last answer, then its question, deleted; both written again.
        assert mgr.delete_last_message(cid, "assistant") and mgr.delete_last_message(cid, "user")
        mgr.add_message(cid, "user", "And service 5?", origin="typed")
        mgr.add_message(cid, "assistant", kept, origin="assistant")
        assert len(_base(mgr, cid)) == len(state.flesh.turns()), "control: the retry leaves the count as it was"
        state.mirror(mgr.get_mirror_messages(cid))
        assert _shape(state.flesh.turns()) == _base(mgr, cid)
        texts = [t["text"] for t in state.flesh.turns()]
        assert rejected not in texts and kept in texts
    finally:
        restore()


def test_mi2_after_a_synchronisation_the_flesh_is_the_conversation_and_the_same_one_again_changes_nothing(tmp_path):
    conv, lib, loaded, restore = _window(tmp_path)
    try:
        mgr, cid = _conversation(conv, tmp_path)
        _say(mgr, cid, 1, 6, typed=True)
        state = lib.state_for(cid)
        state.mirror(mgr.get_mirror_messages(cid))
        # A received turn is legacy whatever it was here: every origin changes, no count does.
        assert mgr.apply_synced_conversation(_payload(mgr, cid))
        assert {o for _r, _t, o, _s in _base(mgr, cid)} == {"legacy"}, "control: the store holds them legacy"
        state.mirror(mgr.get_mirror_messages(cid))
        assert _shape(state.flesh.turns()) == _base(mgr, cid)
        ids = _ids(state)
        assert mgr.apply_synced_conversation(_payload(mgr, cid))
        assert state.mirror(mgr.get_mirror_messages(cid)) == 0, "the same conversation again changes nothing"
        assert _ids(state) == ids
        assert mgr.apply_synced_conversation(_payload(mgr, cid, edit={5: "A peer's answer, regenerated there."}))
        state.mirror(mgr.get_mirror_messages(cid))
        assert _shape(state.flesh.turns()) == _base(mgr, cid)
    finally:
        restore()


# ---------------------------------------------------------------------------
# MI3-MI5 -- a divergence in the Cellar: supersession, never rewriting
# ---------------------------------------------------------------------------
def _cellar_divergence(tmp_path):
    """Eight legacy turns, three spans of two evicted with their peels; two turns left in the Flesh."""
    conv, lib, loaded, restore = _window(tmp_path)
    mgr, cid = _conversation(conv, tmp_path)
    _say(mgr, cid, 1, 8)
    state = lib.state_for(cid)
    state.mirror(mgr.get_mirror_messages(cid))
    _evict(lib, loaded, state, 3)
    return conv, lib, loaded, restore, mgr, cid, state


_EDITED = "An edited fourth turn about service 4."


def test_mi3_a_divergence_in_the_cellar_supersedes_its_span_and_the_later_ones_and_rewrites_nothing(tmp_path):
    conv, lib, loaded, restore, mgr, cid, state = _cellar_divergence(tmp_path)
    try:
        assert [r.turn_ids for r in state.ledger.all()] == [("t0001", "t0002"), ("t0003", "t0004"),
                                                            ("t0005", "t0006")], "control"
        kinds = [r.kind for r in state.ledger.all()]
        before = _cellar_bytes(state)
        assert mgr.apply_synced_conversation(_payload(mgr, cid, edit={3: _EDITED}))
        state.mirror(mgr.get_mirror_messages(cid))
        assert [r.kind for r in state.ledger.all()] == [kinds[0], "superseded", "superseded"]
        assert _cellar_bytes(state) == before, "every span of the Cellar as it was, byte for byte"
    finally:
        restore()


def test_mi4_after_a_divergence_in_the_cellar_the_state_mirrors_the_conversation_again(tmp_path):
    conv, lib, loaded, restore, mgr, cid, state = _cellar_divergence(tmp_path)
    try:
        assert mgr.apply_synced_conversation(_payload(mgr, cid, edit={3: _EDITED}))
        state.mirror(mgr.get_mirror_messages(cid))
        assert _live(state) == _base(mgr, cid)
        flesh = state.flesh.turns()
        assert [t["text"] for t in flesh[:2]] == [_line(3), _EDITED], "written again from the span's first turn"
        assert all(int(t["turn_id"][1:]) > 8 for t in flesh), "under turn ids never given before"
    finally:
        restore()


def test_mi5_a_superseded_span_s_peel_leaves_the_block_and_the_span_stays_recallable(tmp_path):
    conv, lib, loaded, restore, mgr, cid, state = _cellar_divergence(tmp_path)
    try:
        gone = state.ledger.all()[1]
        peel = next(p for p in state.tree.all() if p.sources == (gone.key,))
        question = "service 3 service 4 cluster"
        assert peel.text in lib.memory_block(cid, question, budget=_budget(loaded)), "control: the peel is selected"
        assert mgr.apply_synced_conversation(_payload(mgr, cid, edit={3: _EDITED}))
        state.mirror(mgr.get_mirror_messages(cid))
        assert peel.text not in lib.memory_block(cid, question, budget=_budget(loaded))
        assert lib.recall(cid, gone.key) == state.cellar.get(gone.key), "the user still reads what it was"
    finally:
        restore()


# ---------------------------------------------------------------------------
# MI6-MI7 -- turn ids are never given twice
# ---------------------------------------------------------------------------
def _sqlite_store(lib, loaded, path):
    store_mod = loaded["opti_oignon.memory.onion_store"]
    opener = lambda p: sqlite3.connect(str(p))  # noqa: E731
    lib._store[(str(path), False)] = store_mod.OnionStore(path, connect=opener, require_encryption=False)


def _config(lib, **over):
    fields = dict(enabled=True, model="fake:1b", keep_alive="0", min_new_turns=4, temperature=0.0, num_predict=128)
    fields.update(over)
    return lib.LibrarianConfig(**fields)


def test_mi6_a_turn_id_is_never_given_twice_across_a_save_and_a_load(tmp_path):
    conv, lib, loaded, restore = _window(tmp_path, persisted=True)
    try:
        mgr, cid = _conversation(conv, tmp_path)
        path = tmp_path / "onion.db"
        _sqlite_store(lib, loaded, path)
        config = _config(lib, persist_path=str(path), require_encryption=False)
        _say(mgr, cid, 1, 4, typed=True)
        state = lib.state_for(cid, config)
        state.mirror(mgr.get_mirror_messages(cid))
        given = set(_ids(state))
        assert mgr.delete_last_message(cid, "assistant")
        mgr.add_message(cid, "assistant", "A second answer.", origin="assistant")
        state.mirror(mgr.get_mirror_messages(cid))
        given |= set(_ids(state))
        assert len(given) == 5, "control: the retry gave one id more"
        lib._save_state(cid, state, config)
        lib._states.clear()
        back = lib.state_for(cid, config)
        assert back is not state and _ids(back) == _ids(state), "control: read back from the store"
        assert mgr.delete_last_message(cid, "assistant")
        mgr.add_message(cid, "assistant", "A third answer.", origin="assistant")
        back.mirror(mgr.get_mirror_messages(cid))
        new = set(_ids(back)) - given
        assert len(new) == 1 and not new & given
        assert all(int(i[1:]) > max(int(g[1:]) for g in given) for i in new)
    finally:
        restore()


def test_mi7_a_step_whose_span_was_taken_back_during_its_summary_evicts_nothing(tmp_path):
    conv, lib, loaded, restore = _window(tmp_path)
    try:
        mgr, cid = _conversation(conv, tmp_path)
        mgr.add_message(cid, "user", "Where does service 1 live?", origin="typed")
        mgr.add_message(cid, "assistant", "Service 1 lives on cluster 1 since 2026-03-02.", origin="assistant")
        state = lib.state_for(cid)
        state.mirror(mgr.get_mirror_messages(cid))

        def retried_meanwhile(turns):
            assert mgr.delete_last_message(cid, "assistant")
            mgr.add_message(cid, "assistant", "Service 1 lives on cluster 9 since 2026-03-02.", origin="assistant")
            state.mirror(mgr.get_mirror_messages(cid))
            return _faithful(turns)

        outcome = lib.curate(state, retried_meanwhile, gate=_gate(loaded), budget=_budget(loaded, flesh=1))
        assert outcome.evicted is False and outcome.rung == "stale"
        assert state.ledger.all() == [] and state.tree.all() == [], "no peel stands for words it never read"
        assert _shape(state.flesh.turns()) == _base(mgr, cid)
    finally:
        restore()


# ---------------------------------------------------------------------------
# MI8 -- proposals of a superseded span
# ---------------------------------------------------------------------------
_DECIDED = "We keep Docker on the build server."
_TYPED_TURNS = [
    ("user", "Alice moved the build to Berlin on 2026-03-04. " + _DECIDED, "typed"),
    ("assistant", "Noted: the Berlin build runs 12 jobs a day. Bob checks the logs every morning.", "assistant"),
]
_LOSSY = "Alice moved the build to Berlin on 2026-03-04. The Berlin build runs 12 jobs a day. Bob checks the logs every morning."


def test_mi8_a_proposal_whose_span_is_superseded_is_neither_offered_nor_taken(tmp_path):
    conv, lib, loaded, restore = _window(tmp_path)
    try:
        peels = loaded["opti_oignon.memory.peels"]
        mgr, cid = _conversation(conv, tmp_path)
        for role, text, origin in _TYPED_TURNS:
            mgr.add_message(cid, role, text, origin=origin)
        _say(mgr, cid, 3, 2, typed=True)
        state = lib.state_for(cid)
        state.mirror(mgr.get_mirror_messages(cid))
        gate = replace(peels.load_gate(), span_turns=2)
        ladder = replace(peels.load_ladder(), rho=0.1)
        outcome = lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=_budget(loaded, flesh=1), ladder=ladder)
        assert outcome.rung == "held", "control: the decision is held, not summarised"
        offered = lib.proposals(cid)
        assert [p["text"] for p in offered] == [_DECIDED], "control: the typed decision is offered"
        # A peer's history: the first turn edited there.
        assert mgr.apply_synced_conversation(_payload(mgr, cid, edit={0: "Alice moved the build to Lyon."}))
        state.mirror(mgr.get_mirror_messages(cid))
        assert lib.proposals(cid) == []
        try:
            lib.accept_proposal(cid, offered[0]["id"], actor="user")
        except KeyError as exc:
            assert "not open" in str(exc)
        else:
            raise AssertionError("a superseded proposal was pinned to the Core")
        assert state.core.all() == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# MI9-MI10 -- fewer turns, no turn
# ---------------------------------------------------------------------------
def test_mi9_a_conversation_that_lost_its_last_turns_takes_them_back_from_the_flesh(tmp_path):
    conv, lib, loaded, restore = _window(tmp_path)
    try:
        mgr, cid = _conversation(conv, tmp_path)
        _say(mgr, cid, 1, 6, typed=True)
        state = lib.state_for(cid)
        state.mirror(mgr.get_mirror_messages(cid))
        assert mgr.delete_last_message(cid, "assistant") and mgr.delete_last_message(cid, "user")
        assert state.mirror(mgr.get_mirror_messages(cid)) == 2
        assert _shape(state.flesh.turns()) == _base(mgr, cid)
    finally:
        restore()


def test_mi10_an_empty_conversation_changes_nothing(tmp_path):
    conv, lib, loaded, restore, mgr, cid, state = _cellar_divergence(tmp_path)
    try:
        flesh, receipts, cellar = state.flesh.turns(), state.ledger.all(), _cellar_bytes(state)
        assert state.mirror([]) == 0
        assert state.mirror([{"role": "user", "content": "   "}, "not a message"]) == 0
        assert state.flesh.turns() == flesh and state.ledger.all() == receipts and _cellar_bytes(state) == cellar
    finally:
        restore()


# ---------------------------------------------------------------------------
# MI11-MI12 -- counted and logged without a word
# ---------------------------------------------------------------------------
def test_mi11_the_mirror_counts_appended_taken_back_and_superseded(tmp_path):
    conv, lib, loaded, restore, mgr, cid, state = _cellar_divergence(tmp_path)
    try:
        counted = lib.counters().get("mirror", {})
        assert counted.get("appended") == 8, "the eight turns mirrored"
        mgr.add_message(cid, "user", _line(9))
        state.mirror(mgr.get_mirror_messages(cid))
        assert mgr.delete_last_message(cid, "user")
        state.mirror(mgr.get_mirror_messages(cid))
        assert lib.counters()["mirror"].get("taken_back") == 1, "one Flesh turn taken back"
        assert mgr.apply_synced_conversation(_payload(mgr, cid, edit={0: "An edited first turn."}))
        state.mirror(mgr.get_mirror_messages(cid))
        assert lib.counters()["mirror"].get("superseded") == 3, "the three receipts superseded"
    finally:
        restore()


def test_mi12_what_the_mirror_logs_of_a_divergence_names_turns_and_counts_never_a_word(tmp_path, caplog):
    conv, lib, loaded, restore, mgr, cid, state = _cellar_divergence(tmp_path)
    try:
        canary = "Wolframite-Quokka-7731"
        assert mgr.apply_synced_conversation(_payload(mgr, cid, edit={2: f"The {canary} turn."}))
        with caplog.at_level(logging.DEBUG, logger="opti_oignon.memory.librarian"):
            state.mirror(mgr.get_mirror_messages(cid))
        said = [r.getMessage() for r in caplog.records if r.name == "opti_oignon.memory.librarian"]
        assert any("t0003" in s for s in said), "the divergence is said, by its turn"
        words = {w for t in _base(mgr, cid) for w in t[1].split() if len(w) > 3} | {canary}
        assert not [s for s in said for w in words if w in s], "no word of a turn"
    finally:
        restore()


# ---------------------------------------------------------------------------
# MI13-MI14 -- refusal marks: one at most, the head's
# ---------------------------------------------------------------------------
def _typed_messages():
    return [{"role": r, "content": t, "origin": o, "segments": []} for r, t, o in _TYPED_TURNS]


def test_mi13_after_a_step_a_state_holds_at_most_the_mark_of_the_span_at_the_head():
    _conv, lib, loaded, restore = _window()
    try:
        peels = loaded["opti_oignon.memory.peels"]
        gate, ladder = replace(peels.load_gate(), span_turns=2), peels.load_ladder()
        witness = lib.OnionState()
        witness.mirror(_typed_messages())
        marks = {}
        peels.advance(flesh=witness.flesh, cellar=witness.cellar, ledger=witness.ledger, tree=witness.tree, gate=gate,
                      ladder=ladder, summarize=lambda turns: _LOSSY, reask=lambda turns, missing: _LOSSY, refusals=marks)
        assert len(marks) == 1, "control: the step marks its refusal"
        state = lib.OnionState()
        state.mirror(_typed_messages() * 2)
        state.refusals["f" * 64] = "0" * 64
        outcome = lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=_budget(loaded, flesh=1), ladder=ladder,
                             reask=lambda turns, missing: _LOSSY)
        assert outcome.evicted, "control: the span left"
        assert state.refusals == {}, "the span took its mark with it, and a foreign one is gone"
    finally:
        restore()


class _DiesAtCommit:
    """The state's lock, which the process dies inside at its third use: the step's commit."""

    def __init__(self, lock):
        self._lock, self._uses = lock, 0

    def __enter__(self):
        self._uses += 1
        if self._uses == 3:
            raise RuntimeError("the process died at the commit")
        return self._lock.__enter__()

    def __exit__(self, *exc):
        return self._lock.__exit__(*exc)


def test_mi14_a_mark_kept_with_its_span_at_the_head_spares_the_second_asking():
    _conv, lib, loaded, restore = _window()
    try:
        peels = loaded["opti_oignon.memory.peels"]
        gate, ladder = replace(peels.load_gate(), span_turns=2), peels.load_ladder()
        state = lib.OnionState()
        state.mirror(_typed_messages())
        asked = []

        def reask(turns, missing):
            asked.append(missing)
            return _LOSSY

        try:
            peels.advance(flesh=state.flesh, cellar=state.cellar, ledger=state.ledger, tree=state.tree, gate=gate,
                          ladder=ladder, summarize=lambda turns: _LOSSY, reask=reask, refusals=state.refusals,
                          lock=_DiesAtCommit(state.lock))
        except RuntimeError:
            pass
        assert len(asked) == 1 and len(state.refusals) == 1 and len(state.flesh.turns()) == 2, "control"
        outcome = lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=_budget(loaded, flesh=1), ladder=ladder,
                             reask=reask)
        assert outcome.evicted and len(asked) == 1, "the span kept its mark: not asked for a third summary"
    finally:
        restore()


# ---------------------------------------------------------------------------
# MI15 -- random histories against a reference, and the defect they find
# ---------------------------------------------------------------------------
def _normal(lib_probes, messages):
    out = []
    for m in messages:
        if not str(m.get("content", "") or "").strip():
            continue
        declared = {"role": m["role"], "text": m["content"], "origin": m.get("origin", "legacy"),
                    "segments": m.get("segments", [])}
        origin, segments, _defect = lib_probes.read_origin(declared)
        out.append((m["role"], m["content"], origin, [list(s) for s in segments]))
    return out


def _history(lib, loaded, rng, mirror):
    """One random history; the violations it finds, by name. ``mirror(state, messages)`` mirrors."""
    probes = loaded["opti_oignon.memory.probes"]
    state, messages, made = lib.OnionState(), [], [0]
    cellar, given, violations = {}, {}, []

    def turn(role, typed):
        made[0] += 1
        origin = ("typed" if role == "user" else "assistant") if typed else "legacy"
        return {"role": role, "content": _line(made[0]), "origin": origin, "segments": []}

    for _ in range(rng.randint(8, 18)):
        op = rng.choice(("append", "append", "append", "retry", "delete", "sync", "edit", "step", "step"))
        if op == "append":
            messages += [turn("user", rng.random() < 0.5), turn("assistant", rng.random() < 0.5)]
        elif op == "retry" and messages and messages[-1]["role"] == "assistant":
            messages[-1] = turn("assistant", True)
        elif op == "delete" and messages:
            del messages[-rng.randint(1, min(2, len(messages))):]
        elif op == "sync":
            messages = [dict(m, origin="legacy", segments=[]) for m in messages]
        elif op == "edit" and messages:
            index = rng.randrange(len(messages))
            messages[index] = dict(messages[index], content=_line(10_000 + made[0]))
            made[0] += 1
        elif op == "step" and state.flesh.turns():
            lib.curate(state, _faithful, gate=_gate(loaded), budget=_budget(loaded, flesh=1))
        if messages:
            mirror(state, [dict(m) for m in messages])
            if _live(state) != _normal(probes, messages):
                violations.append("the mirror is not the conversation")
        for key, text in _cellar_bytes(state).items():
            if cellar.setdefault(key, text) != text:
                violations.append("a span of the Cellar changed")
        for t in state.flesh.turns() + [t for r in state.ledger.all() for t in state.cellar.get(r.key)]:
            digest = hashlib.sha256(json.dumps([t["role"], t["text"], t["origin"]]).encode()).hexdigest()
            if given.setdefault(t["turn_id"], digest) != digest:
                violations.append("a turn id given twice")
        gone = [r for r in state.ledger.all() if r.kind == "superseded"]
        if gone:
            lib._states["fuzz"] = state
            block = lib.memory_block("fuzz", "service cluster", budget=_budget(loaded))
            if any(r.key[:12] in block for r in gone):
                violations.append("a superseded receipt reached the block")
    return violations


def _counting_mirror(state, messages):
    """The mirror before identity: it appends what lies beyond the count it has seen."""
    valid = [m for m in messages if str(m.get("content", "") or "").strip()]
    with state.lock:
        for m in valid[state.seen:]:
            state.seen += 1
            state.flesh.append({"turn_id": f"t{state.seen:04d}", "role": m["role"], "text": m["content"],
                                "origin": m.get("origin", "legacy"), "segments": list(m.get("segments", []))})


def test_mi15_random_histories_keep_the_mirror_the_conversation_and_find_a_counting_mirror():
    _conv, lib, loaded, restore = _window()
    try:
        found, planted = [], []
        for seed in range(60):
            found += _history(lib, loaded, random.Random(seed), lambda state, messages: state.mirror(messages))
            planted += _history(lib, loaded, random.Random(seed), _counting_mirror)
        assert found == [], sorted(set(found))
        assert "the mirror is not the conversation" in planted, "witness: the histories find a mirror that counts"
    finally:
        restore()


# ---------------------------------------------------------------------------
# MI16 -- the store keeps what supersession wrote
# ---------------------------------------------------------------------------
def test_mi16_supersession_and_the_ids_given_come_back_from_the_store_as_saved(tmp_path):
    conv, lib, loaded, restore = _window(tmp_path, persisted=True)
    try:
        peels = loaded["opti_oignon.memory.peels"]
        mgr, cid = _conversation(conv, tmp_path)
        path = tmp_path / "onion.db"
        _sqlite_store(lib, loaded, path)
        config = _config(lib, persist_path=str(path), require_encryption=False)
        for role, text, origin in _TYPED_TURNS:
            mgr.add_message(cid, role, text, origin=origin)
        _say(mgr, cid, 3, 2, typed=True)
        state = lib.state_for(cid, config)
        state.mirror(mgr.get_mirror_messages(cid))
        gate, ladder = replace(peels.load_gate(), span_turns=2), replace(peels.load_ladder(), rho=0.1)
        assert lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=_budget(loaded, flesh=1), ladder=ladder).rung == "held"
        assert len(lib.proposals(cid)) == 1, "control: one proposal"
        assert mgr.apply_synced_conversation(_payload(mgr, cid, edit={0: "Alice moved the build to Lyon."}))
        state.mirror(mgr.get_mirror_messages(cid))
        root = lib._save_state(cid, state, config)
        kinds, statuses, seen = [r.kind for r in state.ledger.all()], [q.status for q in state.proposals], state.seen
        assert kinds == ["superseded"] and statuses == ["superseded"], "control: supersession wrote both"
        lib._states.clear()
        back = lib.state_for(cid, config)
        assert [r.kind for r in back.ledger.all()] == kinds
        assert [q.status for q in back.proposals] == statuses
        assert back.seen == seen
        assert lib._save_state(cid, back, config) == root, "the same root over the same rows"
    finally:
        restore()


# ---------------------------------------------------------------------------
# MI17 -- the watermark follows the Flesh taken back
# ---------------------------------------------------------------------------
def test_mi17_once_the_flesh_is_taken_back_a_burst_fires_after_the_growth_from_what_is_left(tmp_path):
    conv, lib, loaded, restore = _window(tmp_path)
    try:
        mgr, cid = _conversation(conv, tmp_path)
        fired = []
        config = _config(lib, min_new_turns=4)
        _say(mgr, cid, 1, 8, typed=True)
        assert lib.maybe_curate(cid, mgr.get_mirror_messages(cid), config=config, runner=fired.append)
        assert fired == [cid], "control: eight turns of growth fire a burst"
        for _ in range(4):
            assert mgr.delete_last_message(cid)
        assert not lib.maybe_curate(cid, mgr.get_mirror_messages(cid), config=config, runner=fired.append)
        _say(mgr, cid, 20, 4, typed=True)
        assert lib.maybe_curate(cid, mgr.get_mirror_messages(cid), config=config, runner=fired.append)
        assert fired == [cid, cid], "four new turns after the Flesh was taken back fire the next burst"
    finally:
        restore()


# ---------------------------------------------------------------------------
# MI18 -- LB7, but for a shorter history
# ---------------------------------------------------------------------------
def _lb_messages(n):
    out = []
    for i in range(1, n + 1):
        role = "user" if i % 2 else "assistant"
        out.append({"role": role, "content": f"Turn {i}: Alice reviewed service {i} on 2026-03-{i:02d} and we agreed that service {i} stays on the new cluster."})
    return out


def test_mi18_the_mirror_is_exact_and_idempotent_and_a_shorter_history_takes_back_what_it_lost():
    _conv, lib, loaded, restore = _window()
    try:
        state = lib.state_for("c1")
        msgs = _lb_messages(5)
        assert state.mirror(msgs) == 5
        turns = state.flesh.turns()
        assert [t["turn_id"] for t in turns] == [f"t{i:04d}" for i in range(1, 6)]
        assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant", "user"]
        assert [t["text"] for t in turns] == [m["content"] for m in msgs]
        assert state.mirror(msgs) == 0 and len(state.flesh.turns()) == 5
        assert state.mirror(msgs + _lb_messages(7)[5:]) == 2 and len(state.flesh.turns()) == 7
        assert state.mirror([{"role": "user"}, {"role": "user", "content": ""}] + msgs) == 2, (
            "a shorter history takes back the two turns it no longer holds; a malformed message is no turn"
        )
        assert [t["text"] for t in state.flesh.turns()] == [m["content"] for m in msgs]
        assert [t["turn_id"] for t in state.flesh.turns()] == [f"t{i:04d}" for i in range(1, 6)], "what stays keeps its id"
    finally:
        restore()


# ---------------------------------------------------------------------------
# MI19-MI22 -- what supersession closes, and the mirror's memory of the state
# ---------------------------------------------------------------------------
def _held_decision(tmp_path):
    """A typed decision held in the Cellar, from the queue's own step, then the first turn edited by a peer."""
    conv, lib, loaded, restore = _window(tmp_path)
    peels = loaded["opti_oignon.memory.peels"]
    mgr, cid = _conversation(conv, tmp_path)
    for role, text, origin in _TYPED_TURNS:
        mgr.add_message(cid, role, text, origin=origin)
    _say(mgr, cid, 3, 2, typed=True)
    state = lib.state_for(cid)
    state.mirror(mgr.get_mirror_messages(cid))
    gate, ladder = replace(peels.load_gate(), span_turns=2), replace(peels.load_ladder(), rho=0.1)
    return conv, lib, loaded, restore, mgr, cid, state, gate, ladder


def test_mi19_a_proposal_left_open_on_a_superseded_span_is_neither_offered_nor_taken(tmp_path):
    conv, lib, loaded, restore, mgr, cid, state, gate, ladder = _held_decision(tmp_path)
    try:
        lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=_budget(loaded, flesh=1), ladder=ladder)
        assert len(lib.proposals(cid)) == 1, "control: offered"
        assert mgr.apply_synced_conversation(_payload(mgr, cid, edit={0: "Alice moved the build to Lyon."}))
        state.mirror(mgr.get_mirror_messages(cid))
        # A file written by another hand: the proposal open again, its span still superseded.
        state.proposals = [replace(q, status="open") for q in state.proposals]
        assert lib.proposals(cid) == []
        try:
            lib.accept_proposal(cid, state.proposals[0].id, actor="user")
        except KeyError:
            pass
        else:
            raise AssertionError("a decision the conversation no longer holds was pinned to the Core")
        assert state.core.all() == []
    finally:
        restore()


def test_mi20_no_proposal_is_made_from_a_receipt_superseded_between_its_step_and_its_offer(tmp_path):
    conv, lib, loaded, restore, mgr, cid, state, gate, ladder = _held_decision(tmp_path)
    try:
        peels = loaded["opti_oignon.memory.peels"]
        outcome = peels.advance(flesh=state.flesh, cellar=state.cellar, ledger=state.ledger, tree=state.tree,
                                gate=gate, ladder=ladder, summarize=lambda turns: _LOSSY, lock=state.lock)
        assert outcome.rung == "held" and state.proposals == [], "control: held, nothing offered yet"
        assert mgr.apply_synced_conversation(_payload(mgr, cid, edit={0: "Alice moved the build to Lyon."}))
        state.mirror(mgr.get_mirror_messages(cid))
        assert lib._propose(state, outcome.receipt, gate, ladder) == (0, 0)
        assert state.proposals == []
    finally:
        restore()


def test_mi21_a_superseded_receipt_leaves_the_digest_and_the_user_s_open_receipts(tmp_path):
    conv, lib, loaded, restore, mgr, cid, state = _cellar_divergence(tmp_path)
    try:
        gone = state.ledger.all()[1]
        assert gone.key in [r.key for r in lib.open_receipts(cid)], "control: open"
        assert gone.key[:12] in lib.memory_block(cid, "service", budget=_budget(loaded)), "control: in the digest"
        assert mgr.apply_synced_conversation(_payload(mgr, cid, edit={3: _EDITED}))
        state.mirror(mgr.get_mirror_messages(cid))
        assert [r.key for r in lib.open_receipts(cid)] == [state.ledger.all()[0].key]
        block = lib.memory_block(cid, "service", budget=_budget(loaded))
        assert gone.key[:12] not in block and state.ledger.all()[0].key[:12] in block
    finally:
        restore()


# ---------------------------------------------------------------------------
# MI34-MI37 -- the divergence in the Cellar, over a conversation received from
# a peer (supersede MI3, MI4, MI12, MI21)
# ---------------------------------------------------------------------------
def _received_cellar_divergence(tmp_path):
    """Eight legacy turns as a peer's copy brought them, three spans of two evicted with their peels; two in the Flesh."""
    conv, lib, loaded, restore = _window(tmp_path)
    mgr, cid = _conversation(conv, tmp_path)
    _say(mgr, cid, 1, 8)
    assert mgr.apply_synced_conversation(_payload(mgr, cid)), "control: the conversation is a peer's copy"
    state = lib.state_for(cid)
    state.mirror(mgr.get_mirror_messages(cid))
    _evict(lib, loaded, state, 3)
    return conv, lib, loaded, restore, mgr, cid, state


def test_mi34_a_divergence_in_the_cellar_of_a_received_conversation_supersedes_its_span_and_rewrites_nothing(tmp_path):
    conv, lib, loaded, restore, mgr, cid, state = _received_cellar_divergence(tmp_path)
    try:
        assert [r.turn_ids for r in state.ledger.all()] == [("t0001", "t0002"), ("t0003", "t0004"),
                                                            ("t0005", "t0006")], "control"
        kinds = [r.kind for r in state.ledger.all()]
        before = _cellar_bytes(state)
        assert mgr.apply_synced_conversation(_payload(mgr, cid, edit={3: _EDITED}))
        state.mirror(mgr.get_mirror_messages(cid))
        assert [r.kind for r in state.ledger.all()] == [kinds[0], "superseded", "superseded"]
        assert _cellar_bytes(state) == before, "every span of the Cellar as it was, byte for byte"
    finally:
        restore()


def test_mi35_after_a_divergence_in_the_cellar_of_a_received_conversation_the_state_mirrors_it_again(tmp_path):
    conv, lib, loaded, restore, mgr, cid, state = _received_cellar_divergence(tmp_path)
    try:
        assert mgr.apply_synced_conversation(_payload(mgr, cid, edit={3: _EDITED}))
        state.mirror(mgr.get_mirror_messages(cid))
        assert _live(state) == _base(mgr, cid)
        flesh = state.flesh.turns()
        assert [t["text"] for t in flesh[:2]] == [_line(3), _EDITED], "written again from the span's first turn"
        assert all(int(t["turn_id"][1:]) > 8 for t in flesh), "under turn ids never given before"
    finally:
        restore()


def test_mi36_what_the_mirror_logs_of_a_divergence_in_a_received_conversation_names_turns_never_a_word(
        tmp_path, caplog):
    conv, lib, loaded, restore, mgr, cid, state = _received_cellar_divergence(tmp_path)
    try:
        canary = "Wolframite-Quokka-7731"
        assert mgr.apply_synced_conversation(_payload(mgr, cid, edit={2: f"The {canary} turn."}))
        with caplog.at_level(logging.DEBUG, logger="opti_oignon.memory.librarian"):
            state.mirror(mgr.get_mirror_messages(cid))
        said = [r.getMessage() for r in caplog.records if r.name == "opti_oignon.memory.librarian"]
        assert any("t0003" in s for s in said), "the divergence is said, by its turn"
        words = {w for t in _base(mgr, cid) for w in t[1].split() if len(w) > 3} | {canary}
        assert not [s for s in said for w in words if w in s], "no word of a turn"
    finally:
        restore()


def test_mi37_a_superseded_receipt_of_a_received_conversation_leaves_the_digest_and_the_open_receipts(tmp_path):
    conv, lib, loaded, restore, mgr, cid, state = _received_cellar_divergence(tmp_path)
    try:
        gone = state.ledger.all()[1]
        assert gone.key in [r.key for r in lib.open_receipts(cid)], "control: open"
        assert gone.key[:12] in lib.memory_block(cid, "service", budget=_budget(loaded)), "control: in the digest"
        assert mgr.apply_synced_conversation(_payload(mgr, cid, edit={3: _EDITED}))
        state.mirror(mgr.get_mirror_messages(cid))
        assert [r.key for r in lib.open_receipts(cid)] == [state.ledger.all()[0].key]
        block = lib.memory_block(cid, "service", budget=_budget(loaded))
        assert gone.key[:12] not in block and state.ledger.all()[0].key[:12] in block
    finally:
        restore()


def test_mi22_a_turn_another_writer_took_back_is_mirrored_again(tmp_path):
    conv, lib, loaded, restore = _window(tmp_path)
    try:
        mgr, cid = _conversation(conv, tmp_path)
        _say(mgr, cid, 1, 4, typed=True)
        state = lib.state_for(cid)
        state.mirror(mgr.get_mirror_messages(cid))
        with state.lock:
            state.flesh.take_back(1)
        assert state.mirror(mgr.get_mirror_messages(cid)) == 1
        assert _shape(state.flesh.turns()) == _base(mgr, cid)
    finally:
        restore()


def test_mi23_the_anchors_of_a_superseded_held_span_leave_the_block(tmp_path):
    conv, lib, loaded, restore, mgr, cid, state, gate, ladder = _held_decision(tmp_path)
    try:
        lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=_budget(loaded, flesh=1), ladder=ladder)
        assert _DECIDED in lib.memory_block(cid, "Docker build server", budget=_budget(loaded)), "control: shown"
        assert mgr.apply_synced_conversation(_payload(mgr, cid, edit={0: "Alice moved the build to Lyon."}))
        state.mirror(mgr.get_mirror_messages(cid))
        assert _DECIDED not in lib.memory_block(cid, "Docker build server", budget=_budget(loaded))
    finally:
        restore()


# ---------------------------------------------------------------------------
# MI24-MI27 -- what the independent review of the mirror found
# ---------------------------------------------------------------------------
def _typed_with(answer=None):
    """The typed decision and its answer, then two turns more; ``answer`` regenerates the first answer."""
    turns = [dict(role=r, content=t, origin=o, segments=[]) for r, t, o in _TYPED_TURNS]
    if answer is not None:
        turns[1]["content"] = answer
    return turns + [dict(role="user", content=_line(3), origin="typed", segments=[]),
                    dict(role="assistant", content=_line(4), origin="assistant", segments=[])]


_REGENERATED = "Noted: the Berlin build runs 13 jobs a day. Bob checks the logs every morning."


def _held_state(lib, loaded, cap=5):
    peels = loaded["opti_oignon.memory.peels"]
    gate = replace(peels.load_gate(), span_turns=2)
    ladder = replace(peels.load_ladder(), rho=0.1, proposals_per_day=cap)
    state = lib.state_for("c1")
    state.mirror(_typed_with())
    return state, gate, ladder


def test_mi24_no_proposal_is_made_from_a_span_superseded_while_it_is_drawn():
    _conv, lib, loaded, restore = _window()
    try:
        peels, probes = loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.probes"]
        state, gate, ladder = _held_state(lib, loaded, cap=1)
        outcome = peels.advance(flesh=state.flesh, cellar=state.cellar, ledger=state.ledger, tree=state.tree,
                                gate=gate, ladder=ladder, summarize=lambda turns: _LOSSY, lock=state.lock)
        assert outcome.rung == "held", "control"
        real, fired = probes.generate_probes, []

        def mirror_meanwhile(span, lexicon=None):
            # The chat thread's mirror lands while the proposal's probes are drawn.
            if not fired:
                fired.append(True)
                state.mirror(_typed_with(_REGENERATED))
            return real(span, lexicon)

        probes.generate_probes = mirror_meanwhile
        try:
            made = lib._propose(state, outcome.receipt, gate, ladder)
        finally:
            probes.generate_probes = real
        assert fired, "control: the mirror landed in between"
        assert made == (0, 0) and state.proposals == []
        assert lib._offered_on(state, lib._today()) == 0, "the day's room untouched"
    finally:
        restore()


def test_mi25_a_decision_made_again_after_its_supersession_is_offered_the_same_day():
    _conv, lib, loaded, restore = _window()
    try:
        state, gate, ladder = _held_state(lib, loaded, cap=1)
        assert lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=_budget(loaded, flesh=1),
                          ladder=ladder).rung == "held"
        assert [p["text"] for p in lib.proposals("c1", ladder=ladder)] == [_DECIDED], "control: offered"
        state.mirror(_typed_with(_REGENERATED))
        assert lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=_budget(loaded, flesh=1),
                          ladder=ladder).rung == "held"
        assert [p["text"] for p in lib.proposals("c1", ladder=ladder)] == [_DECIDED], "offered the same day"
    finally:
        restore()


@pytest.mark.parametrize("verdict", ["declined", "accepted"])
def test_mi26_a_decided_decision_is_not_offered_again_once_its_turn_is_mirrored_again(verdict):
    _conv, lib, loaded, restore = _window()
    try:
        state, gate, ladder = _held_state(lib, loaded)
        assert lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=_budget(loaded, flesh=1),
                          ladder=ladder).rung == "held"
        (offered,) = lib.proposals("c1", ladder=ladder)
        if verdict == "declined":
            lib.decline_proposal("c1", offered["id"], actor="user")
        else:
            lib.accept_proposal("c1", offered["id"], actor="user", budget=_budget(loaded))
        state.mirror(_typed_with(_REGENERATED))
        assert lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=_budget(loaded, flesh=1),
                          ladder=ladder).rung == "held", "control: the turn left again, held"
        assert lib.proposals("c1", ladder=ladder) == []
        assert lib.counters()["proposal"].get("already_decided") == 1
    finally:
        restore()


def test_mi27_a_flesh_row_the_mirror_cannot_read_is_taken_back_and_mirrored_again():
    _conv, lib, loaded, restore = _window()
    try:
        state = lib.OnionState()
        # A Flesh row as a file can hold it: the root does not cover the Flesh.
        state.flesh.append({"turn_id": "t0001", "role": "user", "text": "Hello there.", "origin": "legacy",
                            "segments": [5]})
        state.seen = 1
        conversation = [{"role": "user", "content": "Hello there."}, {"role": "assistant", "content": "Hi."}]
        state.mirror(conversation)
        assert [(t["role"], t["text"], t["segments"]) for t in state.flesh.turns()] == [
            ("user", "Hello there.", []), ("assistant", "Hi.", [])]
        assert all(int(t["turn_id"][1:]) > 1 for t in state.flesh.turns()), "the unreadable row was taken back"
    finally:
        restore()


# ---------------------------------------------------------------------------
# MI28 -- random histories of proposals, and the defects they find
# ---------------------------------------------------------------------------
_POOL = ("We keep Docker on the build server.", "We drop Redis for the session cache.",
         "We choose Postgres for the ledger.")
_HISTORIES = itertools.count()


def _proposal_history(lib, loaded, rng):
    """One random history of decisions and supersessions; the violations of the three invariants, by name."""
    peels = loaded["opti_oignon.memory.peels"]
    composer = loaded["opti_oignon.memory.composer"]
    cap = rng.randint(1, 2)
    gate = replace(peels.load_gate(), span_turns=2)
    ladder = replace(peels.load_ladder(), rho=0.1, proposals_per_day=cap)
    tight = composer.Budget(window=4000, reserve=200, core=1000, receipts=300, peels=800, flesh=1, turn=200)
    day = [1]
    lib._today = lambda: f"2026-10-{day[0]:02d}"
    # A conversation of its own: a seed replayed under a planted defect starts afresh.
    cid = f"fuzz-{rng.random()}-{next(_HISTORIES)}"
    state, messages, violations = lib.state_for(cid), [], []
    # What the user was offered, as the user reads it: each proposal listed,
    # the step and the day it was first listed, its words and its decision.
    # One first listed in the words of an offer superseded since, whatever
    # its day, is that offer made again, not a new one.
    first, again, decided_at, step = {}, {}, {}, [0]

    def place(q):
        # A decision read here, not through the queue's own reading of it.
        turn = next(t for t in state.cellar.get(q.span_key) if t["turn_id"] == q.turn_id)
        return (_turn_identity(turn), q.start, q.stop)

    def listed(offered):
        proposals = {q.id: q for q in state.proposals}
        for p in offered:
            if p["id"] in first:
                continue
            today = lib._today()
            before = next((i for i, (_s, _d, w, _k) in first.items() if w == p["text"]
                           and proposals[i].status == "superseded" and i not in again.values()), None)
            if before is not None:
                again[p["id"]] = before
            first[p["id"]] = (step[0], today, p["text"], place(proposals[p["id"]]))
        return offered

    for _ in range(rng.randint(16, 30)):
        step[0] += 1
        op = rng.choice(("say", "say", "say", "regenerate", "edit", "edit", "step", "step", "step",
                         "decide", "decide", "day"))
        if op == "say":
            n = len(messages)
            messages += [dict(role="user", origin="typed", segments=[],
                              content=f"Alice moved the build to Berlin on 2026-03-{n % 28 + 1:02d}. {rng.choice(_POOL)}"),
                         dict(role="assistant", origin="assistant", segments=[],
                              content=f"Noted: the Berlin build runs {n + 10} jobs a day.")]
        elif op in ("regenerate", "edit") and messages:
            # An edit reaches back into the oldest answers, those most likely in the Cellar.
            older = max(2, len(messages) // 2)
            index = (len(messages) - 1) if op == "regenerate" else rng.randrange(1, older, 2)
            messages[index] = dict(messages[index], content=f"Noted again: the build runs {rng.randint(1, 99)} jobs.")
        elif op == "step" and state.flesh.turns():
            lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=tight, ladder=ladder)
        elif op == "decide":
            offered = listed(lib.proposals(cid, ladder=ladder))
            if offered:
                chosen = rng.choice(offered)["id"]
                decided_at.setdefault(first[chosen][3], step[0])
                # Whatever is first listed from here on comes after the verdict.
                step[0] += 1
                if rng.random() < 0.5:
                    lib.decline_proposal(cid, chosen, actor="user")
                else:
                    lib.accept_proposal(cid, chosen, actor="user", budget=tight)
        elif op == "day":
            day[0] += 1
        if messages:
            state.mirror([dict(m) for m in messages])
        listed(lib.proposals(cid, ladder=ladder))
        with state.lock:
            gone = {r.key for r in state.ledger.all() if r.kind == "superseded"}
            if any(q.status == "open" and q.span_key in gone for q in state.proposals):
                violations.append("an open proposal on a superseded span")
            offers = {}
            for pid, (_s, day_listed, _w, _k) in first.items():
                if pid not in again:
                    offers[day_listed] = offers.get(day_listed, 0) + 1
            if any(n > cap for n in offers.values()):
                violations.append("a day offered more than its cap")
            # A copy offered before the verdict stays an offer of its own; none
            # is offered after it.
            if any(key in decided_at and s > decided_at[key] for s, _d, _w, key in first.values()):
                violations.append("a decided decision offered again from its turn")
    return violations


def _turn_identity(turn):
    return json.dumps([turn["role"], turn["text"], turn.get("origin") or "legacy", turn.get("segments") or []])


def _twins(lib, loaded, cap):
    """The same decision message sent twice, each copy held with its answer: two proposals before any verdict."""
    peels = loaded["opti_oignon.memory.peels"]
    gate = replace(peels.load_gate(), span_turns=2)
    ladder = replace(peels.load_ladder(), rho=0.1, proposals_per_day=cap)
    state = lib.state_for(f"twins-{cap}")
    sent = dict(role="user", origin="typed", segments=[], content=_TYPED_TURNS[0][1])
    state.mirror([sent, dict(role="assistant", origin="assistant", segments=[], content=_TYPED_TURNS[1][1]),
                  sent, dict(role="assistant", origin="assistant", segments=[], content=_REGENERATED)])
    for _ in range(2):
        assert lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=_budget(loaded, flesh=1),
                          ladder=ladder).rung == "held", "control"
    return state, ladder


def test_mi29_a_verdict_holds_back_the_copies_of_its_decision_not_yet_offered():
    _conv, lib, loaded, restore = _window()
    try:
        day = ["2026-10-07"]
        lib._today = lambda: day[0]
        state, ladder = _twins(lib, loaded, 1)
        assert [q.status for q in state.proposals] == ["open", "deferred"], "control: the copy waits for its day"
        (first,) = lib.proposals("twins-1", ladder=ladder)
        lib.decline_proposal("twins-1", first["id"], actor="user")
        assert lib.proposals("twins-1", ladder=ladder) == []
        day[0] = "2026-10-08"
        assert lib.proposals("twins-1", ladder=ladder) == [], "and not offered on a later day"
        assert lib.counters()["proposal"].get("already_decided") == 1
    finally:
        restore()


@pytest.mark.parametrize("segments", [None, 0, "", {}, False], ids=["null", "zero", "text", "mapping", "false"])
def test_mi30_a_flesh_row_whose_segments_are_not_a_list_is_taken_back(segments):
    _conv, lib, loaded, restore = _window()
    try:
        state = lib.OnionState()
        state.flesh.append({"turn_id": "t0001", "role": "user", "text": "We keep Docker on the build server.",
                            "origin": "typed", "segments": segments})
        state.seen = 1
        state.mirror([{"role": "user", "content": "We keep Docker on the build server.", "origin": "typed",
                       "segments": []}])
        assert [(t["turn_id"], t["segments"]) for t in state.flesh.turns()] == [("t0002", [])]
        older = lib.OnionState()
        older.flesh.append({"turn_id": "t0001", "role": "user", "text": "Plain words."})
        older.seen = 1
        assert older.mirror([{"role": "user", "content": "Plain words."}]) == 0, "a row from before segments stands"
    finally:
        restore()


def test_mi31_edits_never_offer_more_in_a_day_than_its_cap():
    _conv, lib, loaded, restore = _window()
    try:
        peels = loaded["opti_oignon.memory.peels"]
        gate = replace(peels.load_gate(), span_turns=2)
        ladder = replace(peels.load_ladder(), rho=0.1, proposals_per_day=1)
        day = ["2026-10-07"]
        lib._today = lambda: day[0]
        state, shown = lib.state_for("edits"), set()
        for decision in _POOL:
            state.mirror([dict(role="user", origin="typed", segments=[],
                               content="Alice moved the build to Berlin on 2026-03-04. " + decision),
                          dict(role="assistant", origin="assistant", segments=[], content=_TYPED_TURNS[1][1])])
            assert lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=_budget(loaded, flesh=1),
                              ladder=ladder).rung == "held", "control"
            shown |= {p["text"] for p in lib.proposals("edits", ladder=ladder)}
        assert shown == {_POOL[0]}, "one distinct decision offered on the day"
        day[0] = "2026-10-08"
        assert [p["text"] for p in lib.proposals("edits", ladder=ladder)] == [_POOL[2]], "the one still held, later"
    finally:
        restore()


def _typed_decision(words):
    return dict(role="user", origin="typed", segments=[], content="Alice moved the build to Berlin on 2026-03-04. " + words)


def _answer(text=_TYPED_TURNS[1][1]):
    return dict(role="assistant", origin="assistant", segments=[], content=text)


def test_mi32_an_offer_from_an_earlier_day_made_again_after_an_edit_keeps_its_place():
    _conv, lib, loaded, restore = _window()
    try:
        peels = loaded["opti_oignon.memory.peels"]
        gate = replace(peels.load_gate(), span_turns=2)
        ladder = replace(peels.load_ladder(), rho=0.1, proposals_per_day=1)
        day = ["2026-10-07"]
        lib._today = lambda: day[0]
        state = lib.state_for("days")

        def step():
            assert lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=_budget(loaded, flesh=1),
                              ladder=ladder).rung == "held", "control"

        conversation = [_typed_decision(_POOL[0]), _answer()]
        state.mirror(conversation)
        step()
        day[0] = "2026-10-08"
        conversation += [_typed_decision(_POOL[1]), _answer()]
        state.mirror(conversation)
        step()
        offered = [(p["text"], p["made_on"]) for p in lib.proposals("days", ladder=ladder)]
        assert offered == [(_POOL[0], "2026-10-07"), (_POOL[1], "2026-10-08")], "control: one offer a day"
        conversation[1] = _answer(_REGENERATED)
        state.mirror(conversation)
        step()
        step()
        assert [(p["text"], p["made_on"]) for p in lib.proposals("days", ladder=ladder)] == offered
    finally:
        restore()


def test_mi33_a_listing_on_a_full_day_draws_the_probes_of_what_it_shows_only():
    _conv, lib, loaded, restore = _window()
    try:
        peels, probes = loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.probes"]
        gate = replace(peels.load_gate(), span_turns=2)
        ladder = replace(peels.load_ladder(), rho=0.1, proposals_per_day=1)
        lib._today = lambda: "2026-10-07"
        state = lib.state_for("full")
        conversation = []
        for words in _POOL:
            conversation += [_typed_decision(words), _answer()]
        state.mirror(conversation)
        for _ in _POOL:
            assert lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=_budget(loaded, flesh=1),
                              ladder=ladder).rung == "held", "control"
        assert [q.status for q in state.proposals] == ["open", "deferred", "deferred"], "control: the day is full"
        real, drawn = probes.generate_probes, []

        def counted(span, lexicon=None):
            drawn.append(len(span))
            return real(span, lexicon)

        probes.generate_probes = counted
        try:
            shown = lib.proposals("full", ladder=ladder)
        finally:
            probes.generate_probes = real
        assert [p["text"] for p in shown] == [_POOL[0]], "control"
        assert len(drawn) == len(shown)
    finally:
        restore()


def test_mi28_random_histories_keep_the_proposals_true_and_find_the_planted_defects():
    _conv, lib, loaded, restore = _window()
    try:
        found = []
        for seed in range(20):
            found += _proposal_history(lib, loaded, random.Random(seed))
        assert found == [], sorted(set(found))
        decided, take_back = lib._decided, lib.OnionState._take_back
        lib._decided = lambda state: set()
        forgot = [v for seed in range(20) for v in _proposal_history(lib, loaded, random.Random(seed))]
        lib._decided = decided
        assert "a decided decision offered again from its turn" in forgot, "witness: the histories find forgotten verdicts"

        def leaves_open(self, same, standing, held, ids):
            proposals = list(self.proposals)
            result = take_back(self, same, standing, held, ids)
            self.proposals = proposals
            return result

        lib.OnionState._take_back = leaves_open
        left = [v for seed in range(20) for v in _proposal_history(lib, loaded, random.Random(seed))]
        lib.OnionState._take_back = take_back
        assert "an open proposal on a superseded span" in left, "witness: the histories find proposals left open"
        offered_on = lib._offered_on
        lib._offered_on = lambda state, day: 0
        uncapped = [v for seed in range(20) for v in _proposal_history(lib, loaded, random.Random(seed))]
        lib._offered_on = offered_on
        assert "a day offered more than its cap" in uncapped, "witness: the histories find a day with no cap"
    finally:
        restore()
