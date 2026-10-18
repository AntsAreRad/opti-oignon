#!/usr/bin/env python3
"""Contracts for what a span the queue holds keeps of an order the user typed: every segment it stands in, whole.

A span no peel answers for is held: its receipt keeps, as anchors, the
places in the Cellar of the fewest typed units -- a sentence or a fenced
block -- answering its probes, and a question that reaches one shows its
words in the memory block. A sentence of an order kept alone showed the
order without what bounds it: "Delete the logs on Atlas." without the "Not
before next week." typed after it -- a bound states no fact, so no probe
asks for it -- or without the "Only if Bob agrees." typed before; and as
standing when the user took it back two turns later. A peel keeps the
user's orders by reference to each whole segment they typed that orders,
and each bare retraction typed after one; an anchor now keeps and shows
them the same way, and never offers one to the Core:

  * HA1 -- AN ORDER IS KEPT WITH WHAT BOUNDS IT: a unit the cover keeps
    that reaches into an order is kept by the whole segment the order was
    typed in -- the bound typed after it, the condition or the sign-off
    typed before it, the retraction typed with it -- and never by its own
    place; the memory block shows the whole segment, never the order alone.
  * HA2 -- ANY OTHER UNIT KEEPS ITS OWN PLACE: a unit of a segment that
    orders nothing, and a fact typed beside an order, keep their own
    places, in span order and once, a segment before the units it holds;
    with no table of directives nothing reads as an order.
  * HA3 -- WHAT THE BLOCK SHOWS OF IT: a segment kept whole is shown by its
    words, its fenced block by its marker, a marker the user typed defanged
    and said; a unit within it is not shown again, and when the cap leaves
    the segment out, the unit is shown alone and the segment counted; a
    place that is neither a typed unit nor a whole segment the user typed
    shows nothing and is counted: a piece of a segment, a document's
    segment, a segment of a turn a peer sent.
  * HA4 -- A DECISION IS OFFERED ONLY WHERE IT ORDERS NOTHING: the Core is
    offered a decision kept beside an order in its own words at its own
    place -- the proposal its own anchor made before -- once when it is
    typed twice, and a decision that reaches into an order is offered
    nothing, whether its anchor is the order's segment or its one sentence:
    alone it would stand in the Core without its bound.
  * HA5 -- AN ORDER TYPED ACROSS SEGMENTS KEEPS EACH OF THEM: an order and
    its bound typed as two segments of one turn are kept as both segments,
    the references a peel makes to them, and shown together, once, though
    the question reaches the order's words alone.
  * HA6 -- AN ORDER IS SHOWN WITH EACH RETRACTION TYPED AFTER IT: whether a
    bare "No." typed in a later turn took the order back or answered
    another question, the order is kept -- a prohibition or a standing rule
    as any -- and shown with the "No.", each part after its turn's marker
    as a peel's references are, one item ranked by the words of all its
    parts, so that no part is shown again on its own; a cap that holds the
    order but not the two shows neither and counts it, a withdrawal that
    reaches the "No." holds the order back too, a gate with no table of its
    own reads the shipped one, and a "No." a peer sent is none.
  * HA7 -- THE BLOCK READS ORDERS WITH THE GATE ITS CALLER HOLDS: the gate
    handed to ``memory_block``, and the one a window's plan resolved, reach
    the anchors' reading; the shipped file is not read again.

The onion's modules come from source through the shared isolation window.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_ha1_an_order_a_held_span_keeps_is_kept_whole_with_what_bounds_it": 5.0,
    "test_ha2_any_other_unit_keeps_its_own_place": 5.0,
    "test_ha3_the_block_shows_a_segment_kept_whole_and_each_unit_within_it_once": 5.0,
    "test_ha4_a_decision_is_offered_only_where_it_orders_nothing": 5.0,
    "test_ha5_an_order_typed_across_segments_keeps_each_of_them": 5.0,
    "test_ha6_an_order_is_shown_with_each_retraction_typed_after_it": 5.0,
    "test_ha7_the_block_reads_orders_with_the_gate_its_caller_holds": 5.0,
}

_MODULES = ("probes", "core_store", "receipts", "composer", "peels", "librarian")
_LIBRARIAN = "opti_oignon.memory.librarian"
_PEELS = "opti_oignon.memory.peels"
_PROBES = "opti_oignon.memory.probes"
_RECEIPTS = "opti_oignon.memory.receipts"
_COMPOSER = "opti_oignon.memory.composer"
_CORE = "opti_oignon.memory.core_store"

_ANSWER = "Noted, Bob checks the logs every morning."
_ORDER, _BOUND = "Delete the logs on Atlas.", "Not before next week."
# What the user typed, and the sentence of its order the cover keeps.
_ORDERS = (
    (_ORDER + " " + _BOUND, _ORDER),
    ("Only if Bob agrees. " + _ORDER, _ORDER),
    ("Bob has to sign off first. Then delete the logs on Atlas.", "Then delete the logs on Atlas."),
    ("Alice lives in Oslo. " + _ORDER, _ORDER),
    (_ORDER + " Wait. No.", _ORDER),
)
_FACT = "Alice lives in Oslo."
_MOVED = "Alice moved the build to Berlin on 2026-03-04."
_KEPT = "We keep Docker on the build server."
_DECISION = "We will use Postgres for Atlas."
_DECIDES = _DECISION + " Delete the old MySQL dumps. Not before Friday."


def _open():
    loaded, restore = isolate(
        targets={f"opti_oignon.memory.{name}": source("memory", f"{name}.py") for name in _MODULES},
        blocked=("opti_oignon.inference_backend", "opti_oignon.db_utils"),
        packages=("opti_oignon.memory",),
    )
    loaded[_LIBRARIAN].reset_librarian()
    loaded[_LIBRARIAN]._today = lambda: "2026-10-06"
    return loaded, restore


def _gate(loaded, span_turns=2):
    return replace(loaded[_PEELS].load_gate(), span_turns=span_turns)


def _ladder(loaded):
    return replace(loaded[_PEELS].load_ladder(), rho=0.5)


def _tiny(loaded):
    """A Flesh cap of one token: every step evicts."""
    return loaded[_COMPOSER].Budget(window=780, reserve=60, core=60, receipts=300, peels=160, flesh=1, turn=60)


def _small(loaded):
    return loaded[_COMPOSER].Budget(window=780, reserve=60, core=60, receipts=300, peels=160, flesh=140, turn=60)


def _message(text, role="user", origin="typed", segments=()):
    return {"role": role, "origin": origin, "segments": [list(s) for s in segments], "content": text}


def _said(text):
    """The user's turn ``text`` and the assistant's answer."""
    return [_message(text), _message(_ANSWER, "assistant", "assistant")]


def _held(loaded, cid, messages, gate=None):
    """The state of ``cid`` once the first span mirrored from ``messages`` left with no summariser, and the step."""
    lib = loaded[_LIBRARIAN]
    state = lib.state_for(cid)
    state.mirror(messages)
    outcome = lib.curate(state, None, gate=gate or _gate(loaded), budget=_tiny(loaded), ladder=_ladder(loaded))
    assert outcome.rung == "held" and outcome.peel is None, ("control: the span is held", outcome)
    return state, outcome


def _kept(state, receipt):
    texts = {t["turn_id"]: t["text"] for t in state.cellar.get(receipt.key)}
    return [texts[turn_id][start:stop] for turn_id, start, stop in receipt.anchors]


# ---------------------------------------------------------------------------
# HA1 -- an order is kept with what bounds it
# ---------------------------------------------------------------------------
def test_ha1_an_order_a_held_span_keeps_is_kept_whole_with_what_bounds_it():
    """HA1: the whole segment an order stands in, where the cover keeps the order's sentence; never it alone."""
    loaded, restore = _open()
    lib, peels, probes = loaded[_LIBRARIAN], loaded[_PEELS], loaded[_PROBES]
    try:
        gate = _gate(loaded)
        for n, (typed, alone) in enumerate(_ORDERS):
            cid = f"c{n}"
            state, outcome = _held(loaded, cid, _said(typed))
            span = state.cellar.get(outcome.receipt.key)
            found = peels._typed_units(span)
            drawn = probes.generate_probes(span, gate.lexicon)
            cover = [found[i].text for i in peels._choose(found, peels._within_reach(drawn, found), _ladder(loaded))]
            assert alone in cover, ("control: the cover keeps the order's own sentence", cover)
            kept = _kept(state, outcome.receipt)
            assert kept[0] == typed and alone not in kept, ("the whole segment, never the order alone", kept)
            block = str(lib.memory_block(cid, "delete the logs on atlas", budget=_small(loaded)))
            assert typed in block and block.count(alone) == 1, ("shown whole, and only so", block)
    finally:
        restore()


# ---------------------------------------------------------------------------
# HA2 -- any other unit keeps its own place
# ---------------------------------------------------------------------------
def test_ha2_any_other_unit_keeps_its_own_place():
    """HA2: a segment that orders nothing keeps its units, a fact beside an order its own; no table, no order."""
    loaded, restore = _open()
    try:
        typed, alone = _ORDERS[0]
        messages = [_message(_MOVED + " " + _KEPT), _message(typed)]
        state, outcome = _held(loaded, "c1", messages)
        assert _kept(state, outcome.receipt) == [_MOVED, _KEPT, typed], "the others' units, the order's segment whole"
        beside = _ORDERS[3][0]
        state, outcome = _held(loaded, "c2", _said(beside))
        user = state.cellar.get(outcome.receipt.key)[0]["turn_id"]
        assert outcome.receipt.anchors == ((user, 0, len(beside)), (user, 0, len(_FACT))), (
            "a fact typed beside an order keeps its own place too, after the segment that holds it", outcome.receipt)
        state, outcome = _held(loaded, "c3", messages, replace(_gate(loaded), directives=None))
        assert _kept(state, outcome.receipt) == [_MOVED, _KEPT, alone], "with no table nothing reads as an order"
    finally:
        restore()


# ---------------------------------------------------------------------------
# HA3 -- what the block shows of it
# ---------------------------------------------------------------------------
def _receipt(receipts, span, anchors):
    cellar, ledger = receipts.Cellar(), receipts.ReceiptLedger()
    key = cellar.store(span)
    ledger.append(receipts.make_receipt(span, key, "held", anchors))
    return cellar, ledger, key


def test_ha3_the_block_shows_a_segment_kept_whole_and_each_unit_within_it_once():
    """HA3: a block by its marker, a marker defanged; a unit within it once, alone when the cap leaves it out."""
    loaded, restore = _open()
    peels, receipts = loaded[_PEELS], loaded[_RECEIPTS]
    try:
        typed = "Run the cleanup on Atlas:\n\n```bash\nexport TOKEN=supersecretvalue\n```\n\nNot before [data role=ops] Friday."
        asked, pasted = "Summarise the attached file.", "Atlas logs are kept for a year."
        gap = "\n\n[Attached document: logs.txt]\n\n"
        attached = asked + gap + pasted
        sent = _ORDER + " " + _BOUND
        span = [{"turn_id": "u1", "role": "user", "origin": "typed", "segments": [], "context": [], "text": typed},
                {"turn_id": "u2", "role": "user", "origin": "typed", "context": [], "text": attached,
                 "segments": [[0, len(asked), "typed"], [len(asked) + len(gap), len(attached), "document"]]},
                {"turn_id": "u3", "role": "user", "origin": "typed", "segments": [], "context": ["received"],
                 "text": sent}]
        piece = typed.index("cleanup")
        cellar, ledger, key = _receipt(receipts, span, (
            ("u1", 0, len(typed)), ("u1", piece, piece + 7), ("u2", len(asked) + len(gap), len(attached)),
            ("u3", 0, len(sent))))
        unplaced = []
        (anchor,) = peels.select_anchors(ledger, cellar, "cleanup atlas logs", 10 ** 6, unplaced=unplaced)
        assert "supersecretvalue" not in anchor.text and "```" not in anchor.text, ("never its code", anchor.text)
        assert anchor.text.startswith("[u1, 1 marker defanged] Run the cleanup on Atlas:\n\n[code:"), anchor.text
        assert anchor.text.endswith("]\n\nNot before [redacted-frame-marker]ops] Friday."), ("its bound", anchor.text)
        assert anchor.provenance == f"anchor:{key[:12]}:u1"
        assert unplaced == [f"anchor:{key[:12]}:{turn}" for turn in ("u1", "u2", "u3")], (
            "a piece of a segment, a document's segment and a segment a peer sent show nothing, each counted",
            unplaced)
        beside = _ORDERS[3][0]
        span = [{"turn_id": "u1", "role": "user", "origin": "typed", "segments": [], "context": [], "text": beside}]
        cellar, ledger, key = _receipt(receipts, span, (("u1", 0, len(beside)), ("u1", 0, len(_FACT))))
        shown = peels.select_anchors(ledger, cellar, "alice oslo atlas", 10 ** 6)
        assert [a.text for a in shown] == [beside], ("the segment, and its fact not again", shown)
        dropped, cap = [], peels.estimate_tokens(beside) - 1
        assert peels.estimate_tokens(_FACT) <= cap, "control: the fact fits where the segment does not"
        shown = peels.select_anchors(ledger, cellar, "alice oslo atlas", cap, dropped=dropped)
        assert [a.text for a in shown] == [_FACT], ("the fact alone, never the order", shown)
        assert dropped == [f"anchor:{key[:12]}:u1"], ("the segment left out is counted", dropped)
    finally:
        restore()


# ---------------------------------------------------------------------------
# HA4 -- a decision is offered only where it orders nothing
# ---------------------------------------------------------------------------
def test_ha4_a_decision_is_offered_only_where_it_orders_nothing():
    """HA4: beside an order, offered at its own place and once; reaching into an order, not offered."""
    loaded, restore = _open()
    lib, core = loaded[_LIBRARIAN], loaded[_CORE]
    try:
        state, outcome = _held(loaded, "c1", _said(_DECIDES))
        span = state.cellar.get(outcome.receipt.key)
        user = span[0]["turn_id"]
        whole, own = (user, 0, len(_DECIDES)), (user, 0, len(_DECISION))
        assert outcome.receipt.anchors == (whole, own), ("control: the segment whole, the decision too", outcome)
        (offered,) = lib.proposals("c1")
        assert (offered["text"], offered["turn_id"], offered["origin"]) == (_DECISION, user, "typed"), offered
        (proposal,) = state.proposals
        assert (proposal.start, proposal.stop) == own[1:], "at its own place"
        assert proposal.id == core.proposal_id(outcome.receipt.key, *own), "the proposal its own anchor made"
        twice = _DECISION + " Delete the old MySQL dumps. " + _DECISION
        state, outcome = _held(loaded, "c2", _said(twice))
        assert [(q.start, q.stop) for q in state.proposals] == [(0, len(_DECISION))], (
            "a decision typed twice is offered once, from its first sentence", state.proposals)
        alone = "Decision: delete the logs on Atlas."
        for n, ordering in enumerate((alone + " Not before Friday.", alone)):
            cid = f"o{n}"
            state, outcome = _held(loaded, cid, _said(ordering))
            span = state.cellar.get(outcome.receipt.key)
            decisions = [p.answer for p in loaded[_PROBES].generate_probes(span, _gate(loaded).lexicon)
                         if p.kind == "decision"]
            assert decisions == [alone], ("control: a typed decision", decisions)
            assert _kept(state, outcome.receipt) == [ordering], "control: kept whole, its bound with it"
            assert state.proposals == [] and lib.proposals(cid) == [], ("a decision that orders is offered nothing",
                                                                         ordering)
    finally:
        restore()


# ---------------------------------------------------------------------------
# HA5 -- an order typed across segments keeps each of them
# ---------------------------------------------------------------------------
def test_ha5_an_order_typed_across_segments_keeps_each_of_them():
    """HA5: an order and its bound typed as two segments of one turn: both kept, as a peel references them."""
    loaded, restore = _open()
    lib, peels, probes = loaded[_LIBRARIAN], loaded[_PEELS], loaded[_PROBES]
    try:
        text = _ORDER + " " + _BOUND
        segments = ((0, len(_ORDER), "typed"), (len(_ORDER) + 1, len(text), "typed"))
        state, outcome = _held(loaded, "c1", [_message(text, segments=segments),
                                              _message(_ANSWER, "assistant", "assistant")])
        span = state.cellar.get(outcome.receipt.key)
        user = span[0]["turn_id"]
        both = ((user, 0, len(_ORDER)), (user, len(_ORDER) + 1, len(text)))
        assert outcome.receipt.anchors == both, ("each segment the order stands in", outcome.receipt)
        gate = _gate(loaded)
        refs = peels._with_orders(span, probes.generate_probes(span, gate.lexicon), "", gate)[1]
        assert tuple(ref[:3] for ref in refs) == both, ("the references a peel makes to them", refs)
        for query in ("delete the logs on atlas next week", "delete the logs on atlas"):
            block = str(lib.memory_block("c1", query, budget=_small(loaded)))
            assert f"[{user}] {_ORDER}\n[{user}] {_BOUND}" in block, ("shown together", query, block)
            assert block.count(_BOUND) == 1, ("once", block)
    finally:
        restore()


# ---------------------------------------------------------------------------
# HA6 -- an order is shown with each retraction typed after it
# ---------------------------------------------------------------------------
def test_ha6_an_order_is_shown_with_each_retraction_typed_after_it():
    """HA6: "No." typed after an order, whatever it answered: the order kept and shown with it, never alone."""
    loaded, restore = _open()
    lib, peels, probes, receipts = loaded[_LIBRARIAN], loaded[_PEELS], loaded[_PROBES], loaded[_RECEIPTS]
    try:
        gate = _gate(loaded, span_turns=3)
        cases = (
            (_ORDER, "Shall I delete them now?", "No.", "delete the logs on atlas"),
            ("Never push to the main branch of Atlas.", "Shall I also reformat the tests?", "No.",
             "push to the main branch of atlas"),
            ("Do not delete the logs on Atlas.", "Shall I restart the build server on Berlin now?", "No.",
             "can i delete the logs on atlas"),
            (_ORDER, "Shall I delete them now?", "No. Archive the dumps on Atlas first.", "delete the logs on atlas"),
        )
        for n, (order, question, answer, query) in enumerate(cases):
            cid = f"c{n}"
            state, outcome = _held(loaded, cid, [_message(order), _message(question, "assistant", "assistant"),
                                                 _message(answer)], gate)
            span = state.cellar.get(outcome.receipt.key)
            first, last = span[0]["turn_id"], span[2]["turn_id"]
            if n == 0:
                texts = {t["turn_id"]: t["text"] for t in span}
                refs = peels._with_orders(span, probes.generate_probes(span, gate.lexicon), "", gate)[1]
                assert [texts[at][begin:end] for at, begin, end, _digest in refs] == [_ORDER, "No."], (
                    "control: a peel references the order and the retraction both", refs)
            assert (first, 0, len(order)) in outcome.receipt.anchors, ("the order kept, a rule as any", outcome)
            block = str(lib.memory_block(cid, query, budget=_small(loaded)))
            assert f"[{first}] {order}\n[{last}] {answer}" in block, ("shown with what was typed after it", block)
            assert block.count(order) == 1, ("and only so", block)
        coded = "Delete the logs on Atlas:\n\n```bash\nrm -rf /var/log/atlas\n```\n\nNo."
        state, outcome = _held(loaded, "c9", _said(coded))
        user = state.cellar.get(outcome.receipt.key)[0]["turn_id"]
        assert outcome.receipt.anchors[0] == (user, 0, len(coded)), ("control: the turn kept whole", outcome.receipt)
        block = str(lib.memory_block("c9", "delete the logs on atlas", budget=_small(loaded)))
        assert block.count("Delete the logs on Atlas:") == 1 and "rm -rf" not in block, (
            "a retraction typed in the order's own segment is shown once, with it", block)
        later = "No. Archive the dumps on Kiev first."
        state, outcome = _held(loaded, "c8", [_message(_ORDER), _message("Shall I delete them now?", "assistant",
                                                                         "assistant"), _message(later)], gate)
        assert len(outcome.receipt.anchors) == 2, ("control: the later segment is an anchor of its own", outcome)
        block = str(lib.memory_block("c8", "archive the dumps on kiev and the logs", budget=_small(loaded)))
        assert block.count("Archive the dumps on Kiev first.") == 1 and _ORDER in block, (
            "the item of the order ranks first, and the later segment is not shown again on its own", block)
        rule = "Never push to the main branch of Atlas."
        question = {"turn_id": "a1", "role": "assistant", "origin": "assistant", "segments": [], "context": [],
                    "text": "Shall I also reformat the tests?"}
        span = [{"turn_id": "u1", "role": "user", "origin": "typed", "segments": [], "context": [], "text": rule},
                question,
                {"turn_id": "u2", "role": "user", "origin": "typed", "segments": [], "context": [], "text": "No."}]
        cellar, ledger, key = _receipt(receipts, span, (("u1", 0, len(rule)),))
        (both,) = peels.select_anchors(ledger, cellar, "push main branch atlas", 10 ** 6)
        assert both.text == f"[u1] {rule}\n[u2] No.", ("as one, each part after its turn's marker", both.text)
        dropped, cap = [], peels.estimate_tokens(both.text) - 1
        assert peels.estimate_tokens(rule) <= cap, "control: the order alone would fit"
        assert peels.select_anchors(ledger, cellar, "push main branch atlas", cap, dropped=dropped) == [], (
            "never the order without the retraction")
        assert dropped == [f"anchor:{key[:12]}:u1"], ("left out, and counted", dropped)
        source = "document:" + "d" * 64
        held_back = []
        assert peels.select_anchors(ledger, cellar, "push main branch atlas", 10 ** 6, lineage={"u2": [source]},
                                    withdrawn=(source,), reached=held_back) == [], "a withdrawn No. holds it back"
        assert held_back == [f"anchor:{key[:12]}:u1"], ("and it is counted", held_back)
        bare = replace(peels.load_gate(), directives=None)
        (read,) = peels.select_anchors(ledger, cellar, "push main branch atlas", 10 ** 6, gate=bare)
        assert read.text == both.text, ("a gate with no table reads the shipped one", read.text)
        span[2] = dict(span[2], context=["received"])
        cellar, ledger, key = _receipt(receipts, span, (("u1", 0, len(rule)),))
        assert [a.text for a in peels.select_anchors(ledger, cellar, "push main branch atlas", 10 ** 6)] == [rule], (
            "a No. a peer sent is shown nowhere and takes nothing back")
    finally:
        restore()



# ---------------------------------------------------------------------------
# HA7 -- the block reads orders with the gate its caller holds
# ---------------------------------------------------------------------------
def test_ha7_the_block_reads_orders_with_the_gate_its_caller_holds():
    """HA7: memory_block's gate and the plan's reach the anchors' reading; the shipped file is not read again."""
    import _onion_journal as oj

    loaded, restore = oj.open_window()
    lib, peels = loaded[oj.LIBRARIAN], loaded[oj.PEELS]
    try:
        gate, cfg = replace(peels.load_gate(), span_turns=3), oj.config(lib)
        rule = "Never push to the main branch of Atlas."
        messages = [oj.typed(rule), oj.answer("Shall I also reformat the tests?"), oj.typed("No.")]
        messages += oj.conversation(4, start=4)
        state = lib.state_for("c1", cfg)
        state.mirror(messages)
        first = state.flesh.turns()[0]["turn_id"]
        oj.hold(state, 3, anchors=((first, 0, len(rule)),))
        shown = f"[{first}] {rule}\n[t0003] No."

        def refuse(*_args, **_kwargs):
            raise RuntimeError("the shipped file was read again")

        peels.load_gate = refuse
        block = str(lib.memory_block("c1", "push to the main branch of atlas", gate=gate, config=cfg))
        assert shown in block, ("the block reads with the gate it is handed", block)
        window = oj.window(loaded, "c1", "Can I push to the main branch of Atlas?", messages=messages, config=cfg,
                           gate=gate)
        assert getattr(window, "refusal", None) is None, window
        assert shown in "\n".join(str(message[1]) for message in window.messages), (
            "and a window with the gate its plan resolved", window.messages)
    finally:
        restore()
