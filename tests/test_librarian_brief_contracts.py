#!/usr/bin/env python3
"""Contracts for what the librarian is told beside a span: where the user's words its peel shows stand, and its room.

Measured on the maintainer's machine, a 4B librarian wrote about 136 words
for spans of 316 where the user's words a peel shows as written took 280:
its summary told the whole span again, and two in three were refused for
their length. The words shown stay the doctrine's -- each whole segment the
user typed that holds an order -- and what changes is what the model is
given and told:

  * BF1 -- THE QUEUE TELLS IT: a librarian that takes a brief (``briefed``)
    is handed, with the turns, the places of the words the peel's
    references will show -- each whole segment the user typed that orders
    -- and the room: the most words a summary may write for the peel to
    stay within its bound, the words shown and their markers counted. On
    the queue's step the bound is the tighter of the gate's length bound and
    the ladder's compression floor (``rho``), a target: a summary of that
    many words keeps the peel within the floor, counted in the gate's words,
    and one word more does not, the bound compared as a ratio (0.29 of 100
    words leaves 29, not 28). With no bound, or no word in the span, there
    is no room to tell. A summariser that takes no brief is called with the
    turns alone, as before.
  * BF2 -- THE SECOND ASKING TOO: a reasker that takes a brief gets the
    first asking's own, after the facts the first summary lost; one that
    does not gets those facts alone. A second refusal under another floor
    is marked apart, and asked again.
  * BF3 -- THE MODEL IS NOT GIVEN THOSE WORDS: the librarian's call leaves
    out a turn the shown words fill and cuts a shown segment from its turn,
    the rest of the turn -- a pasted part -- kept; a place that is no
    segment of its turn cuts nothing. Its instructions say the words are
    left out when some were, hold the budget -- the share ``budget_share``
    of the room, rounded down, one word at least -- and ask for plain prose
    naming who said each thing. A span the shown words leave nothing of, or
    no room, asks the model nothing and is summarised by nothing. Without a
    brief, the request is the one before. The second asking is made the
    same way, the lost facts on its last line, and its identity, which
    marks a refusal, is the prompt it sends with its share: a span refused
    under the old request, or another share, is asked again.
  * BF4 -- EVERY ASKING: a leaf, a parent and a gated eviction brief the
    librarian as the queue's step does.
  * BF5 -- THE SHARE COMES FROM ONION.YAML: the shipped file states
    ``budget_share`` and the loader reads it there; a file without it is
    refused by name, and a share outside (0, 1] -- 0, above 1, a boolean, a
    string, NaN -- is refused.
  * BF6 -- NO ROOM, NO ASKING: when the user's words the peel would show
    already pass the ladder's floor, the queue asks a librarian that takes
    a brief nothing: no summary could make that peel save enough, and a
    peel of those words alone, accepted for the gate's looser bound, would
    stand in the window in place of what saves more. The step goes to the
    repair, which holds the span with its anchors, or rebuilds the peel from
    the user's words when that saves enough in tokens, the floor's own
    measure; either way the step says so, and the queue counts it by why
    (``brief``, ``no_room``), no refusal counted. A summariser that takes no
    brief is asked as before.
  * BF7 -- THE RUNBOOK MEASURES THE LIBRARIAN'S OWN REQUEST: the runbook's
    summariser, a run of its own per call, takes the brief and hands it to
    the librarian; every call its measurement makes is briefed, at the
    gate's bound as a gated eviction briefs; and its second face and sweep
    judge a summary beside the references its peel would make.

The onion's modules come from source through the shared isolation window;
the model is a recording stand-in, and the librarian's own call runs
through its governed run with no governor, as on a machine without one.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_bf1_the_queue_tells_a_briefed_librarian_what_the_peel_shows_and_the_room_it_leaves": 5.0,
    "test_bf2_the_second_asking_carries_the_first_askings_brief_after_the_lost_facts": 5.0,
    "test_bf3_the_librarians_call_leaves_out_the_words_shown_and_holds_its_budget": 5.0,
    "test_bf4_a_leaf_a_parent_and_a_gated_eviction_brief_the_librarian_too": 5.0,
    "test_bf5_the_budget_share_comes_from_onion_yaml_and_is_held_to_a_share[0]": 5.0,
    "test_bf5_the_budget_share_comes_from_onion_yaml_and_is_held_to_a_share[0.0]": 5.0,
    "test_bf5_the_budget_share_comes_from_onion_yaml_and_is_held_to_a_share[1.5]": 5.0,
    "test_bf5_the_budget_share_comes_from_onion_yaml_and_is_held_to_a_share[True]": 5.0,
    "test_bf5_the_budget_share_comes_from_onion_yaml_and_is_held_to_a_share[0.5]": 5.0,
    "test_bf5_the_budget_share_comes_from_onion_yaml_and_is_held_to_a_share[nan]": 5.0,
    "test_bf6_a_span_whose_shown_words_pass_the_floor_asks_a_briefed_librarian_nothing": 5.0,
    "test_bf7_the_runbook_measures_the_librarians_own_request": 5.0,
}

_MODULES = ("probes", "core_store", "receipts", "composer", "peels", "librarian", "baseline")
_LIBRARIAN = "opti_oignon.memory.librarian"
_PEELS = "opti_oignon.memory.peels"
_PROBES = "opti_oignon.memory.probes"
_RECEIPTS = "opti_oignon.memory.receipts"
_COMPOSER = "opti_oignon.memory.composer"
_SHIPPED_ONION = REPO / "opti_oignon" / "config" / "onion.yaml"

_ORDER = "Always answer in French. We keep Docker on the build server."
_ANSWER = "Noted: the Berlin build runs 12 jobs a day. Bob checks the logs every morning."
_SUMMARY = "The Berlin build runs 12 jobs a day. Bob checks the logs every morning."
_LOSSY = "Bob checks the logs every morning."
_LONG = (_ANSWER + " The farm keeps 16 Go per host. The nightly backup runs at 02:00 on Atlas. Mallory reviews "
         "the alerts on Mondays.")


def _open():
    targets = {f"opti_oignon.memory.{name}": source("memory", f"{name}.py") for name in _MODULES}
    loaded, restore = isolate(targets=targets, blocked=("opti_oignon.inference_backend", "opti_oignon.db_utils",
                                                        "opti_oignon.resource_governor"),
                              packages=("opti_oignon.memory",))
    loaded[_LIBRARIAN].reset_librarian()
    return loaded, restore


def _exchange(user=_ORDER, answer=_ANSWER):
    """One exchange as the store hands it to the mirror."""
    return [{"role": "user", "origin": "typed", "segments": [], "content": user, "context": [], "lineage": []},
            {"role": "assistant", "origin": "assistant", "segments": [], "content": answer, "context": [],
             "lineage": []}]


def _turn(turn_id, role, origin, text):
    return {"turn_id": turn_id, "role": role, "origin": origin, "text": text}


def _gate(loaded):
    return replace(loaded[_PEELS].load_gate(), span_turns=2)


def _tiny(loaded):
    """A Flesh cap of one token: every step evicts."""
    return loaded[_COMPOSER].Budget(window=780, reserve=60, core=60, receipts=300, peels=160, flesh=1, turn=60)


def _shipped_share():
    return yaml.safe_load(_SHIPPED_ONION.read_text(encoding="utf-8"))["librarian"]["budget_share"]


def _config(lib, **over):
    fields = dict(enabled=True, model="fake:1b", keep_alive="5m", min_new_turns=4, temperature=0.0, num_predict=64,
                  num_ctx=8192, window_margin=0.25, run_budget_s=300.0, call_timeout_s=5.0)
    fields.update(over)
    return lib.LibrarianConfig(**fields)


class _Backend:
    """A backend that records the system prompt and the content of each call, and answers one sentence."""

    def __init__(self):
        self.calls = []

    def generate(self, model, messages, options=None, keep_alive="30m", think=False, images=None):
        self.calls.append((messages[0]["content"], messages[1]["content"]))
        return types.SimpleNamespace(content="Bob checks the logs every morning.")


def _length_refused(peels, probes, gate, span, words):
    """Whether a summary of ``words`` words is refused for its length beside the span's references to its orders."""
    found = probes.generate_probes(span, gate.lexicon)
    _text, refs = peels._with_orders(span, found, "", gate)
    decision = peels.decide(span, found, " ".join(["jobs"] * words), gate, refs)
    return "length" in peels.refusal_motives(decision, gate)


# ---------------------------------------------------------------------------
# BF1 -- the queue tells it
# ---------------------------------------------------------------------------
def test_bf1_the_queue_tells_a_briefed_librarian_what_the_peel_shows_and_the_room_it_leaves():
    """BF1: the brief holds the segment that orders and the room, exact at the queue's bound."""
    loaded, restore = _open()
    lib, peels, probes = loaded[_LIBRARIAN], loaded[_PEELS], loaded[_PROBES]
    try:
        gate, ladder = _gate(loaded), peels.load_ladder()
        assert ladder.rho < gate.max_length_ratio, ("control: the ladder's floor is the tighter bound", ladder)
        heard = []

        def summarize(turns, brief):
            heard.append(([dict(t) for t in turns], brief))
            return _SUMMARY

        summarize.briefed = True
        state = lib.state_for("c1")
        state.mirror(_exchange(answer=_LONG))
        lib.curate(state, summarize, gate=gate, budget=_tiny(loaded))
        assert len(heard) == 1, heard
        span, brief = heard[0]
        user = next(t["turn_id"] for t in span if t["role"] == "user")
        assert brief.shown == ((user, 0, len(_ORDER)),), ("where the peel's words stand: the segment that orders", brief)
        assert isinstance(brief.room, int) and brief.room > 0, ("control: there is room to tell", brief)
        found = probes.generate_probes(span, gate.lexicon)
        _text, refs = peels._with_orders(span, found, "", gate)
        assert [ref[:3] for ref in refs] == list(brief.shown), "control: the places are the references the peel makes"

        def ratio(words):
            return peels.decide(span, found, " ".join(["jobs"] * words), gate, refs).length_ratio

        assert ratio(brief.room) <= ladder.rho < ratio(brief.room + 1), \
            ("the room keeps the peel within the ladder's floor, one word more does not", brief, ladder.rho)
        assert peels.brief(span, found, gate).room > brief.room, "the gate's bound alone leaves a looser room"
        assert peels.brief(span, found, gate, rho=2.0).room == peels.brief(span, found, gate).room, \
            "the tighter of the two bounds"
        assert peels._room(100, 0, 0.29) == 29, "the bound compared as a ratio: 29 words of 100 are within 0.29"
        assert peels._room(100, 30, 0.29) == 0, "words shown past the bound leave no room, never less"
        assert peels.brief(span, found, replace(gate, max_length_ratio=None)).room is None, "no bound, no room"
        assert peels.brief(span, found, replace(gate, max_length_ratio=None), ladder.rho).room == brief.room, \
            "the floor alone bounds the room where the gate holds no bound"
        assert peels.brief([], found, gate) == peels.Brief((), None), "no word, no room"
        assert peels.brief(span, found, replace(gate, max_length_ratio=1e308)).room is None, \
            "a bound too wide to count words against leaves no room to tell, and breaks nothing"
        typed = "Run this:\n\n```bash\ndocker compose up -d\n```"
        coded = [_turn("u1", "user", "typed", typed), _turn("a1", "assistant", "assistant", "Done.")]
        told = peels.brief(coded, probes.generate_probes(coded, gate.lexicon), gate)
        assert told.shown == (("u1", 0, len(typed)),), ("the whole segment, its block with it", told)
        marker = probes.code_marker("docker compose up -d")
        shown = peels.shown_words(coded, [("u1", 0, len(typed), peels.segment_digest(typed))])
        assert shown == [("u1", "Run this:\n\n" + marker)], ("control: shown with its block as its marker", shown)
        span_words = sum(probes.word_count(text) for text in probes.holdings(coded, gate.directives).texts)
        assert told.room == peels._room(span_words, probes.word_count("[u1] Run this:\n\n" + marker),
                                        gate.max_length_ratio), ("the room counts the block by its marker", told)
        plain = lib.state_for("c2")
        plain.mirror(_exchange())
        unbriefed = lib.curate(plain, lambda turns: _SUMMARY, gate=gate, budget=_tiny(loaded))
        assert unbriefed.rung == "accepted", ("a summariser that takes no brief is called as before", unbriefed)
    finally:
        restore()


# ---------------------------------------------------------------------------
# BF2 -- the second asking too
# ---------------------------------------------------------------------------
def test_bf2_the_second_asking_carries_the_first_askings_brief_after_the_lost_facts():
    """BF2: a briefed reasker gets the lost facts and the same brief; another, the facts alone."""
    loaded, restore = _open()
    lib = loaded[_LIBRARIAN]
    try:
        gate = _gate(loaded)
        first, again = [], []

        def summarize(turns, brief):
            first.append(brief)
            return _LOSSY

        def reask(turns, missing, brief):
            again.append((list(missing), brief))
            return _SUMMARY

        summarize.briefed = reask.briefed = True
        state = lib.state_for("c1")
        state.mirror(_exchange())
        outcome = lib.curate(state, summarize, gate=gate, budget=_tiny(loaded), reask=reask)
        assert outcome.rung == "reasked", ("control: the first summary lost a fact, the second is accepted", outcome)
        assert len(first) == 1 and len(again) == 1, (first, again)
        missing, brief = again[0]
        assert missing, "control: the reasker is handed the facts the first summary lost"
        assert brief == first[0], ("the first asking's brief, word for word", brief, first[0])
        told = []
        plain = lib.state_for("c2")
        plain.mirror(_exchange())
        unbriefed = lib.curate(plain, lambda turns: _LOSSY, gate=gate, budget=_tiny(loaded),
                               reask=lambda turns, missing: told.append(list(missing)) or _SUMMARY)
        assert unbriefed.rung == "reasked" and told and told[0] == missing, ("the facts alone, as before", unbriefed)
        peels, receipts = loaded[_PEELS], loaded[_RECEIPTS]
        span = [_turn("u1", "user", "typed", _ORDER), _turn("a1", "assistant", "assistant", _LONG)]

        def lossy(turns, brief):
            return _LOSSY

        def still_lossy(turns, missing, brief):
            return _LOSSY

        lossy.briefed = still_lossy.briefed = True
        still_lossy.identity = "the librarian's second asking"
        marks = []
        for rho in (0.5, 0.4):
            refusals = {}
            peels.advance(flesh=receipts.Flesh(span + [_turn("u9", "user", "typed", "Thanks.")]),
                          cellar=receipts.Cellar(), ledger=receipts.ReceiptLedger(), tree=peels.PeelTree(), gate=gate,
                          ladder=replace(peels.load_ladder(), rho=rho), summarize=lossy, reask=still_lossy,
                          refusals=refusals)
            marks.append(refusals)
        assert all(len(m) == 1 for m in marks), ("control: each second refusal is marked", marks)
        assert list(marks[0].values()) != list(marks[1].values()), "a refusal under another floor is another mark"
    finally:
        restore()


# ---------------------------------------------------------------------------
# BF3 -- the model is not given those words
# ---------------------------------------------------------------------------
def test_bf3_the_librarians_call_leaves_out_the_words_shown_and_holds_its_budget():
    """BF3: the turns given, the instructions with the budget, nothing asked when nothing is left, the reask's identity."""
    loaded, restore = _open()
    lib, peels = loaded[_LIBRARIAN], loaded[_PEELS]
    try:
        share = _shipped_share()
        assert isinstance(share, float) and 0 < share < 1, ("control: the shipped share is a share below 1", share)
        backend = _Backend()
        config = _config(lib, budget_share=share)
        run = lib._Run(config, governance=False)
        summarize = lib.registry_summarizer(config, resolve=lambda model: backend, run=run)
        reask = lib.registry_reasker(config, resolve=lambda model: backend, run=run)
        assert summarize.briefed is True and reask.briefed is True, "both take the queue's brief"
        user, answer = _turn("t0001", "user", "typed", _ORDER), _turn("t0002", "assistant", "assistant", _ANSWER)
        turns = [user, answer]
        brief = peels.Brief(shown=(("t0001", 0, len(_ORDER)),), room=41)
        budget = math.floor(41 * share)
        assert budget < 41, "control: the budget is a share of the room, not the room"
        summarize(turns, brief)
        system, content = backend.calls[-1]
        assert content == lib._quoted([answer]), ("the turn the shown words fill is left out", content)
        assert _ORDER not in content and _ORDER not in system, "the model never reads the words the peel shows"
        assert system == (lib._SYSTEM_PROMPT + lib._LEFT_OUT_RULE + lib._BRIEF_RULE
                          + lib._BUDGET_RULE.format(budget=f"{budget} words")), system
        for said in ("left out of the turns you are given", f"must not exceed {budget} words.", "plain prose",
                     "who said"):
            assert said in system, said
        summarize(turns, peels.Brief((), 20))
        assert backend.calls[-1] == (lib._SYSTEM_PROMPT + lib._BRIEF_RULE
                                     + lib._BUDGET_RULE.format(budget=f"{math.floor(20 * share)} words"),
                                     lib._quoted(turns)), "nothing shown: every turn given, nothing said left out"
        typed, pasted = "Wipe the old images on Atlas.", "The images are kept in /srv/old.\n\n```bash\nrm -rf /srv/old\n```"
        gap, empty = "\n\n---\nDocument provided: notes.md\n", "\n\n---\nDocument provided: empty.txt\n"
        glued = {"turn_id": "t0003", "role": "user", "origin": "typed", "text": typed + gap + pasted + empty,
                 "segments": [[0, len(typed), "typed"], [len(typed) + len(gap), len(typed) + len(gap) + len(pasted),
                                                         "document"]]}
        given = json.loads(lib._quoted([glued], (("t0003", 0, len(typed)),)))["text"]
        marker = loaded[_PROBES].code_marker("rm -rf /srv/old")
        assert given == gap + "The images are kept in /srv/old.\n\n" + marker + empty, \
            ("the shown segment cut; the executor's lines and the pasted part kept, its block as its marker", given)
        assert json.loads(lib._quoted([user], (("t0001", 0, 5),)))["text"] == _ORDER, \
            "a place that is no segment of its turn cuts nothing"
        calls = len(backend.calls)
        assert summarize([user], brief) == "" and len(backend.calls) == calls, "nothing left: no call, no summary"
        assert summarize(turns, peels.Brief(brief.shown, 0)) == "" and len(backend.calls) == calls, \
            "no room: no call, no summary"
        summarize(turns, peels.Brief(brief.shown, 1))
        assert backend.calls[-1][0].endswith("must not exceed 1 word."), "one word at least while there is room"
        summarize(turns, peels.Brief(brief.shown, None))
        assert backend.calls[-1][0] == lib._SYSTEM_PROMPT + lib._LEFT_OUT_RULE + lib._BRIEF_RULE, \
            "no bound: no budget to state"
        summarize(turns)
        assert backend.calls[-1] == (lib._SYSTEM_PROMPT, lib._quoted(turns)), "without a brief, the request before"
        calls = len(backend.calls)
        assert reask(turns, [("episodic", "12", "t0002")], peels.Brief(brief.shown, 0)) == "" \
            and reask([user], [("episodic", "12", "t0002")], brief) == "" and len(backend.calls) == calls, \
            "no room, or nothing left: the second asking makes no call either"
        reask(turns, [("episodic", "12", "t0002")], brief)
        system, content = backend.calls[-1]
        head, _, last = content.rpartition("\n")
        assert system == (lib._SYSTEM_PROMPT + lib._LEFT_OUT_RULE + lib._BRIEF_RULE
                          + lib._BUDGET_RULE.format(budget=f"{budget} words") + lib._REASK_RULE), system
        assert head == lib._quoted([answer]), "the second asking leaves the shown words out too"
        assert json.loads(last) == {"must_keep": [{"kind": "episodic", "fact": "12", "turn": "t0002"}]}, last
        reask(turns, [("episodic", "12", "t0002")])
        system, content = backend.calls[-1]
        assert system == lib._SYSTEM_PROMPT + lib._REASK_RULE and content.rpartition("\n")[0] == lib._quoted(turns), \
            "without a brief, the second asking before"
        assert reask.identity == lib._call_identity(
            config, lib._SYSTEM_PROMPT + lib._LEFT_OUT_RULE + lib._BRIEF_RULE + lib._BUDGET_RULE + lib._REASK_RULE
            + f" (budget_share {share!r})"), "a refusal is marked with the prompt the reask sends and its share"
        other = lib.registry_reasker(replace(config, budget_share=0.5), resolve=lambda model: backend, run=run)
        assert other.identity != reask.identity, "another share is another request"
        assert reask.identity != lib._call_identity(config, lib._SYSTEM_PROMPT + lib._REASK_RULE), \
            "a span refused under the old request is asked again"
    finally:
        restore()


# ---------------------------------------------------------------------------
# BF4 -- every asking
# ---------------------------------------------------------------------------
def test_bf4_a_leaf_a_parent_and_a_gated_eviction_brief_the_librarian_too():
    """BF4: build_leaf, build_parent and evict_gated hand a briefed summariser its brief."""
    loaded, restore = _open()
    peels, probes, receipts = loaded[_PEELS], loaded[_PROBES], loaded[_RECEIPTS]
    try:
        gate = _gate(loaded)
        span = [_turn("u1", "user", "typed", _ORDER), _turn("a1", "assistant", "assistant", _ANSWER)]
        heard = []

        def summarize(turns, brief):
            # The answer's span is told; the other's words are all shown, and its summary says them once.
            heard.append(brief)
            return _SUMMARY if any(t.get("text") == _ANSWER for t in turns) else "Wipe the old images on Atlas."

        summarize.briefed = True
        expected = peels.brief(span, probes.generate_probes(span, gate.lexicon), gate)
        assert expected.shown == (("u1", 0, len(_ORDER)),) and expected.room, ("control: there is something to tell",
                                                                              expected)
        flesh = receipts.Flesh(span + [_turn("u9", "user", "typed", "Thanks.")])
        gated = peels.evict_gated(flesh=flesh, cellar=receipts.Cellar(), ledger=receipts.ReceiptLedger(),
                                  tree=peels.PeelTree(), gate=gate, summarize=summarize)
        assert gated.evicted is True and heard == [expected], ("the gated eviction briefs it", gated, heard)
        cellar, tree = receipts.Cellar(), peels.PeelTree()
        other = [_turn("u2", "user", "typed", "Wipe the old images on Atlas."), _turn("a2", "assistant", "assistant",
                                                                                     "Done.")]
        leaf, _decision = peels.build_leaf(cellar.store(span), cellar, summarize, gate, tree)
        second, _decision = peels.build_leaf(cellar.store(other), cellar, summarize, gate, tree)
        assert leaf is not None and second is not None, "control: both leaves are accepted"
        assert heard[1] == expected, ("the leaf briefs it", heard[1])
        both = span + other
        peels.build_parent((leaf.id, second.id), cellar, summarize, gate, tree)
        assert len(heard) == 4 and heard[3] == peels.brief(both, probes.generate_probes(both, gate.lexicon), gate), \
            ("the parent briefs it over its children's spans", heard)
        assert len(heard[3].shown) == 2, ("both segments that order", heard[3])
    finally:
        restore()


# ---------------------------------------------------------------------------
# BF5 -- the share comes from onion.yaml
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0, 0.0, 1.5, True, "0.5", float("nan")])
def test_bf5_the_budget_share_comes_from_onion_yaml_and_is_held_to_a_share(tmp_path, bad):
    """BF5: ``budget_share`` is read from the file, required, and refused outside (0, 1]."""
    loaded, restore = _open()
    lib = loaded[_LIBRARIAN]
    try:
        shipped = _SHIPPED_ONION.read_text(encoding="utf-8")
        assert "\n  budget_share:" in shipped, "control: the shipped file states it"
        whole = tmp_path / "onion.yaml"
        whole.write_text(shipped, encoding="utf-8")
        cfg = lib.load_config(whole)
        assert cfg.validate() == [] and cfg.budget_share == _shipped_share(), cfg
        changed = tmp_path / "changed.yaml"
        changed.write_text(shipped.replace(f"\n  budget_share: {_shipped_share()}", "\n  budget_share: 0.5"),
                           encoding="utf-8")
        assert lib.load_config(changed).budget_share == 0.5, "read from the file, not from a default"
        omitted = tmp_path / "omitted.yaml"
        omitted.write_text("\n".join(line for line in shipped.splitlines() if not line.startswith("  budget_share:")),
                           encoding="utf-8")
        with pytest.raises(lib.LibrarianError, match="budget_share"):
            lib.load_config(omitted)
        assert any("budget_share" in error for error in replace(cfg, budget_share=bad).validate()), bad
        assert replace(cfg, budget_share=1).validate() == [], "a share of 1 asks for the whole room"
    finally:
        restore()


# ---------------------------------------------------------------------------
# BF6 -- no room, no asking
# ---------------------------------------------------------------------------
_RULES = ("Always answer in French, keep every reply under five sentences, cite the file and the line you rely on, "
          "and never run a command on the build server before Bob has read the plan.")
_COMPOUNDS = "Always deploy build-server/alpha-one to rack-two/slot-three before Mallory runs qa-run/beta-four."
_PLAIN = "Sure, I will take care of that today and let you know how it goes once it is done."


def test_bf6_a_span_whose_shown_words_pass_the_floor_asks_a_briefed_librarian_nothing():
    """BF6: shown words past ``rho`` of the span: no asking, the repair, a hold; an unbriefed summariser still asked."""
    loaded, restore = _open()
    lib, peels, probes = loaded[_LIBRARIAN], loaded[_PEELS], loaded[_PROBES]
    try:
        gate, ladder = _gate(loaded), peels.load_ladder()
        asked = []

        def summarize(turns, brief):
            asked.append(brief)
            return "The assistant agreed."

        summarize.briefed = True
        state = lib.state_for("c1")
        state.mirror(_exchange(user=_RULES, answer="Agreed."))
        span = [dict(t) for t in state.flesh.turns()[: gate.span_turns]]
        told = peels.brief(span, probes.generate_probes(span, gate.lexicon), gate, ladder.rho)
        assert told.shown and told.room == 0, ("control: the shown words pass the floor", told)
        counted = lib.counters().get("brief", {}).get("no_room", 0)
        outcome = lib.curate(state, summarize, gate=gate, budget=_tiny(loaded))
        assert asked == [], ("no model is asked for a summary that could not save enough", asked)
        assert outcome.rung == "held" and outcome.peel is None, ("the span is held with its anchors", outcome)
        assert outcome.unasked == "no_room" and outcome.refused == (), ("said, and no refusal counted", outcome)
        assert lib.counters().get("brief", {}).get("no_room", 0) == counted + 1, "and counted by why"
        assert outcome.receipt is not None and outcome.receipt.anchors, ("its anchors stay", outcome.receipt)
        # The floor is counted in the gate's words, the repair in tokens: an order written in compounds leaves no
        # room by the first and saves enough by the second, and the repair makes the peel, still unasked.
        compact = [_turn("u1", "user", "typed", _COMPOUNDS), _turn("a1", "assistant", "assistant", _PLAIN)]
        assert peels.brief(compact, probes.generate_probes(compact, gate.lexicon), gate, ladder.rho).room == 0, \
            "control: no room in the gate's words"
        repaired = peels.advance(
            flesh=loaded[_RECEIPTS].Flesh(compact + [_turn("u9", "user", "typed", "Thanks.")]),
            cellar=loaded[_RECEIPTS].Cellar(), ledger=loaded[_RECEIPTS].ReceiptLedger(), tree=peels.PeelTree(),
            gate=gate, ladder=ladder, summarize=summarize)
        assert asked == [] and repaired.rung == "repaired" and repaired.unasked == "no_room", \
            ("rebuilt from the user's words, the model unasked, and said", repaired)
        heard = []
        plain = lib.state_for("c2")
        plain.mirror(_exchange(user=_RULES, answer="Agreed."))
        lib.curate(plain, lambda turns: heard.append(turns) or "The assistant agreed.", gate=gate,
                   budget=_tiny(loaded))
        assert len(heard) == 1, "a summariser that takes no brief is asked as before"
    finally:
        restore()


# ---------------------------------------------------------------------------
# BF7 -- the runbook measures the same request
# ---------------------------------------------------------------------------
def _runbook():
    """The runbook script, loaded from its file; the path it adds is taken back."""
    spec = importlib.util.spec_from_file_location("oo_onion_runbook_bf", REPO / "scripts" / "onion_runbook.py")
    module = importlib.util.module_from_spec(spec)
    path = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = path
    return module


def test_bf7_the_runbook_measures_the_librarians_own_request():
    """BF7: the runbook's summariser takes the brief and hands it on; every measured call is briefed."""
    loaded, restore = _open()
    lib, peels = loaded[_LIBRARIAN], loaded[_PEELS]
    try:
        runbook = _runbook()
        backend = _Backend()
        config = _config(lib, budget_share=_shipped_share())
        summarize = runbook._each_call_a_run(config, resolve=lambda model: backend)
        assert summarize.briefed is True, "the queue's brief reaches the runbook's summariser"
        turns = [_turn("t0001", "user", "typed", _ORDER), _turn("t0002", "assistant", "assistant", _ANSWER)]
        summarize(turns, peels.Brief((("t0001", 0, len(_ORDER)),), 41))
        assert backend.calls[-1][1] == lib._quoted(turns[1:]), "and through it, the librarian's own request"
        told = []

        def recording(turns, *brief):
            told.append(brief)
            return " ".join(t["text"] for t in turns)

        recording.briefed = True
        runbook._measure(recording, config)
        assert told and all(len(b) == 1 and isinstance(b[0], peels.Brief) for b in told), \
            ("every call the measurement makes is briefed", told[:3])
        gate, probes = peels.load_gate(), loaded[_PROBES]
        found = probes.generate_probes(turns, gate.lexicon)
        ordered, refs = peels._with_orders(turns, found, _SUMMARY, gate)
        assert refs, "control: the peel references the order"
        judged = runbook._decisions([(turns, found, ordered, refs)], gate)
        assert judged[0].length_ratio == peels.decide(turns, found, ordered, gate, refs).length_ratio > \
            peels.decide(turns, found, ordered, gate).length_ratio, \
            "its second face and its sweep judge the summary beside the peel's references, as the queue does"
    finally:
        restore()
