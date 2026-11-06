"""Contracts for how the anchors a question reaches are ranked: by BM25 over their words (block A).

The anchors of held receipts were ranked by the share of the question's
terms their words hold: an anchor holding one rare word of the question
ranked below one holding two of its common words, and the evidence of a
question about a long conversation fell out of the cap more often than not
(LoCoMo replayed offline, ten conversations). They are now ranked by BM25
over the anchors the question reaches, the reach, the cap and the order of
what is shown unchanged.

  * AR1 -- THE RAREST WORD FIRST: an anchor holding the one word of the
    question no other holds ranks before anchors holding more of its common
    words, laid where the cap keeps one; the ranker sees only the anchors
    the question reaches, and the score shown is the ranker's.
  * AR2 -- A SEGMENT STILL BEFORE WHAT IT HOLDS: BM25 scores a short fact
    above the whole segment it stands in; the segment, which holds an
    order, still ranks first and its fact is not shown again; the fact
    alone where the cap leaves the segment out, the segment counted.
  * AR3 -- A UNIT THAT ONLY ASKS STILL AFTER: an earlier question holding
    more of the question's words than the fact that answers it ranks after
    the fact, as with the share.
  * AR4 -- THE SHARE WHERE SQLITE HAS NO FTS5: the ranker is None, said once
    in the log, tried once; the anchors rank by the share of the terms.
  * AR5 -- NOTHING WRITTEN, NO QUESTION KEPT: ranking opens SQLite in memory
    only and, as Python sees its opens, no file for writing.
  * AR6 -- THE WINDOW AND THE BLOCK RANK SO: the memory block and a composed
    window lay the anchor BM25 ranks first where the cap keeps one, and the
    anchor the share ranks first when the ranker is None.
  * AR11 -- A RANKER THAT FAILS IS NONE: one that raises under the lock
    leaves the share of the terms, said by name; the window is composed.
  * AR12 -- THE SHARE BREAKS A TIE: anchors a ranker scores alike are
    ordered by the share of the question's terms their words hold, then by
    age, as before.
  * AR13 -- THE QUESTION'S TERMS, NOT ITS FUNCTION WORDS: BM25 is asked the
    terms that reach (a name of two letters among them), never "what",
    "did" or "about": the anchor that answers is laid where the cap keeps
    one, as the share lays it.
  * AR14 -- A TEXT COUNTED ONCE: a sentence several receipts keep is one row
    of the table, so the rare word it holds stays rare.
  * AR15 -- THE NEWEST FIRST AMONG EQUALS: anchors holding the same terms of
    the question tie whatever their lengths; the newest receipt's is laid,
    a fact's update before the stale fact.
  * AR16 -- EQUALS ARE THE ANCHORS THAT HOLD THE SAME TERMS: every anchor
    holding the rare term ranks before every anchor holding only a common
    one, never pooled with them; a cap that keeps one lays one of the first.
  * AR17 -- AN ANCHOR'S OWN WORDS ONLY: a newer anchor whose turn holds a
    term of the question the anchor does not ("I live near the station")
    never takes that term, nor passes the fact and its update.
  * AR18 -- EQUALS NEVER PASS A SEGMENT: a newer anchor holding the terms
    of a segment's fact does not put that fact before its segment, nor
    show it twice.
  * AR19 -- A TERM COUNTS BY ITS STEM: "lives" holds "live": a fact and its
    update rank before anchors that only name the person, the update first
    though older anchors follow it.
  * AR21 -- BELOW THE FLOOR, THE TERMS HELD DECIDE: where every term an
    anchor holds is held by half the anchors or more (FTS5 then gives them
    almost no weight), the stemmed terms it holds order it, then the share
    and the age -- never its length.
  * AR22 -- A FACT INSIDE ITS SEGMENT COUNTED ONCE: a fact kept alone and
    within its segment is one place of its words, so the rare term it holds
    stays rare and the segment is laid before anchors of a common term.
  * AR20 -- ANY SCRIPT FOLDED ALIKE: a Cyrillic name with a letter the
    probes fold ("Maik" written with a short i) matches its turn, as the share finds it.

The onion's modules and the wrapper come from source through the shared
isolation window (``_onion_journal``).
"""
from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _onion_journal as oj  # noqa: E402

BUDGET_S = {
    "test_ar1_the_rarest_word_first": 5.0,
    "test_ar2_a_segment_still_before_what_it_holds": 5.0,
    "test_ar3_a_unit_that_only_asks_still_after": 5.0,
    "test_ar4_the_share_where_sqlite_has_no_fts5": 5.0,
    "test_ar5_nothing_written_no_question_kept": 5.0,
    "test_ar6_the_window_and_the_block_rank_so": 10.0,
    "test_ar11_a_ranker_that_fails_is_none": 10.0,
    "test_ar12_the_anchors_own_words_break_a_tie": 5.0,
    "test_ar13_the_questions_terms_not_its_function_words": 5.0,
    "test_ar14_a_text_counted_once": 5.0,
    "test_ar15_the_newest_first_among_equals": 5.0,
    "test_ar16_equals_are_the_anchors_that_hold_the_same_terms": 5.0,
    "test_ar17_an_anchors_own_words_only": 5.0,
    "test_ar18_equals_never_pass_a_segment": 5.0,
    "test_ar19_a_term_counts_by_its_stem": 5.0,
    "test_ar20_any_script_folded_alike": 5.0,
    "test_ar21_below_the_floor_the_terms_held_decide": 5.0,
    "test_ar22_a_fact_inside_its_segment_counted_once": 5.0,
}

_RECEIPTS = "opti_oignon.memory.receipts"
_RARE = "Caroline walks her greyhound every morning."
_COMMON = ("Caroline met Melanie in the park.", "Melanie and Caroline like the park.",
           "Caroline told Melanie about the park.", "The park was full; Caroline waved at Melanie.")
_OTHER = "Bob repaired the boiler on Tuesday."
_QUESTION = "Did Caroline walk the greyhound in the park with Melanie?"
_ORDER = "Delete the logs on Atlas."
_FACT = "Alice lives in Oslo."
_BESIDE = _FACT + " " + _ORDER


def _span(texts):
    return [{"turn_id": f"u{n}", "role": "user", "origin": "typed", "segments": [], "context": [], "text": text}
            for n, text in enumerate(texts, 1)]


def _receipt(receipts, span, anchors):
    cellar, ledger = receipts.Cellar(), receipts.ReceiptLedger()
    key = cellar.store(span)
    ledger.append(receipts.make_receipt(span, key, "held", anchors))
    return cellar, ledger, key


def _whole(span):
    return tuple((turn["turn_id"], 0, len(turn["text"])) for turn in span)


def test_ar1_the_rarest_word_first():
    """AR1: the anchor holding "greyhound" alone ranks first by BM25, last by the share; the ranker sees what the
    question reaches."""
    loaded, restore = oj.open_window()
    peels, receipts = loaded[oj.PEELS], loaded[_RECEIPTS]
    try:
        span = _span((_RARE,) + _COMMON + (_OTHER,))
        cellar, ledger, _key = _receipt(receipts, span, _whole(span))
        share = peels.select_anchors(ledger, cellar, _QUESTION, 10 ** 6)
        assert share[-1].text == _RARE, ("control: by the share, the rare word's anchor ranks last", share)
        seen = []

        def bm25(query, texts, contexts):
            seen.extend(texts)
            return peels.anchor_bm25(query, texts, contexts)

        ranked = peels.select_anchors(ledger, cellar, _QUESTION, 10 ** 6, rank=bm25)
        assert ranked[0].text == _RARE, ("the rarest word first", ranked)
        assert sorted(seen) == sorted((_RARE,) + _COMMON), ("only what the question reaches", seen)
        given = dict(zip(seen, peels.anchor_bm25(_QUESTION, seen)))
        assert all(item.score == round(given[item.text], 4) for item in ranked), ("its own score", ranked, given)
        cap = peels.estimate_tokens(_RARE)
        assert [a.text for a in peels.select_anchors(ledger, cellar, _QUESTION, cap, rank=peels.anchor_bm25)] == [
            _RARE], "laid where the cap keeps one"
        assert [a.text for a in peels.select_anchors(ledger, cellar, _QUESTION, cap)] != [_RARE], (
            "control: the share lays another")
    finally:
        restore()


def test_ar2_a_segment_still_before_what_it_holds():
    """AR2: a ranker that prefers the short fact; the segment holding the order still first, the fact not again;
    the fact alone where the cap leaves the segment out. (BM25 makes the two equals, as they hold the same terms:
    the rule is for any ranker.)"""
    loaded, restore = oj.open_window()
    peels, receipts = loaded[oj.PEELS], loaded[_RECEIPTS]
    try:
        span = _span((_BESIDE,))
        cellar, ledger, key = _receipt(receipts, span, (("u1", 0, len(_BESIDE)), ("u1", 0, len(_FACT))))
        query = "where does alice live in oslo"
        def pieces(query, texts, contexts):
            return [1.0 / len(text) for text in texts]      # the shorter, the higher

        fact, segment = pieces(query, [_FACT, _BESIDE], None)
        assert fact > segment, ("control: the ranker scores the short fact above its segment", fact, segment)
        shown = peels.select_anchors(ledger, cellar, query, 10 ** 6, rank=pieces)
        assert [a.text for a in shown] == [_BESIDE], ("the segment, and its fact not again", shown)
        dropped, cap = [], peels.estimate_tokens(_BESIDE) - 1
        shown = peels.select_anchors(ledger, cellar, query, cap, dropped=dropped, rank=pieces)
        assert [a.text for a in shown] == [_FACT], ("the fact alone, never the order", shown)
        assert dropped == [f"anchor:{key[:12]}:u1"], ("the segment left out is counted", dropped)
    finally:
        restore()


def test_ar3_a_unit_that_only_asks_still_after():
    """AR3: an earlier question holds more of the words asked than the fact; by BM25 too, the fact first."""
    loaded, restore = oj.open_window()
    peels, receipts = loaded[oj.PEELS], loaded[_RECEIPTS]
    try:
        asked, fact = "Quick check: where does Bob live now, in Lyon?", "Bob lives in Lyon."
        span = _span((asked, fact))
        cellar, ledger, _key = _receipt(receipts, span, _whole(span))
        query = "Where does Bob live now?"
        echo, answer = peels.anchor_bm25(query, [asked, fact])
        assert echo > answer, ("control: BM25 scores the earlier question higher", echo, answer)
        ranked = peels.select_anchors(ledger, cellar, query, 10 ** 6, rank=peels.anchor_bm25)
        assert [a.text for a in ranked] == [fact, asked], ("the fact, then what only asks", ranked)
    finally:
        restore()


def test_ar4_the_share_where_sqlite_has_no_fts5(monkeypatch, caplog):
    """AR4: an SQLite with no FTS5: no ranker, said once, tried once; the share ranks."""
    loaded, restore = oj.open_window()
    peels = loaded[oj.PEELS]
    real = sqlite3.connect
    tries = []

    class NoFts5:
        def __init__(self, db):
            self._db = db

        def execute(self, sql, *args):
            if "fts5" in sql:
                tries.append(sql)
                raise sqlite3.OperationalError("no such module: fts5")
            return self._db.execute(sql, *args)

        def close(self):
            self._db.close()

    try:
        monkeypatch.setattr(peels, "_FTS5", [])
        monkeypatch.setattr(sqlite3, "connect", lambda *a, **kw: NoFts5(real(*a, **kw)))
        with caplog.at_level(logging.WARNING):
            assert peels.anchor_ranker() is None, "no ranker"
            assert peels.anchor_ranker() is None
        said = [r.getMessage() for r in caplog.records if "anchor rank" in r.getMessage()]
        assert len(said) == 1 and "no FTS5" in said[0], ("said once, by name", said)
        assert len(tries) == 1, ("tried once", tries)
        monkeypatch.setattr(sqlite3, "connect", real)
        monkeypatch.setattr(peels, "_FTS5", [])
        assert peels.anchor_ranker() is peels.anchor_bm25, "control: this SQLite holds FTS5"
    finally:
        restore()


def test_ar5_nothing_written_no_question_kept():
    """AR5: a ranking opens SQLite in memory only and opens no file for writing."""
    loaded, restore = oj.open_window()
    peels = loaded[oj.PEELS]
    events, watching = [], [False]

    def hook(event, args):
        # An audit hook stays for the process: it records only while this contract ranks.
        if watching[0] and (event == "sqlite3.connect" or (event == "open" and len(args) > 1 and args[1] and any(
                flag in str(args[1]) for flag in ("w", "a", "x", "+")))):
            events.append((event, args[:2]))

    try:
        sys.addaudithook(hook)
        watching[0] = True
        try:
            peels.anchor_bm25(_QUESTION, (_RARE,) + _COMMON)
        finally:
            watching[0] = False
        connects = [args for event, args in events if event == "sqlite3.connect"]
        writes = [args for event, args in events if event == "open"]
        assert connects and all(str(a[0]) == ":memory:" for a in connects), ("in memory only", connects)
        assert not writes, ("no file opened for writing", writes)
    finally:
        restore()


def _held(lib, cid, units, cfg):
    """A conversation whose first spans, each a typed unit and an answer, were held, each unit its anchor."""
    messages = []
    for unit in units:
        messages += [oj.typed(unit), oj.answer("Noted.")]
    messages += oj.conversation(6, start=len(messages) + 1)
    state = lib.state_for(cid, cfg)
    state.mirror(messages)
    for unit in units:
        oj.hold(state, 2, anchors=((state.flesh.turns()[0]["turn_id"], 0, len(unit)),))
    return state, messages


def test_ar6_the_window_and_the_block_rank_so(monkeypatch):
    """AR6: where the cap keeps one, the block and the window lay the anchor BM25 ranks first; the share's first
    when there is no ranker. A window composed curates its conversation: each reading has its own."""
    loaded, restore = oj.open_window()
    lib, peels, composer = loaded[oj.LIBRARIAN], loaded[oj.PEELS], loaded[oj.COMPOSER]
    try:
        cfg = oj.config(lib)
        units = list(_COMMON * 6) + [_RARE]
        budget = composer.Budget(window=780, reserve=60, core=60, receipts=300, peels=peels.estimate_tokens(_RARE),
                                 flesh=140, turn=60)

        def laid(cid):
            _state, messages = _held(lib, cid, units, cfg)
            block = str(lib.memory_block(cid, _QUESTION, budget=budget))
            win = oj.window(loaded, cid, _QUESTION, messages=messages, config=cfg, model_window=4096,
                            budgets=oj.snug(loaded, _QUESTION))
            assert win.refusal is None, win.refusal
            return block, win.messages[-1][1]

        block, window = laid("c1")
        assert _RARE in block, ("the block lays the rarest word's anchor", block)
        assert _RARE in window, ("the window too", window)
        monkeypatch.setattr(peels, "anchor_ranker", lambda: None)
        block, window = laid("c2")
        assert _RARE not in block and any(c in block for c in _COMMON), ("control: the share lays another", block)
        assert _RARE not in window, ("control: the window's cap binds", window)
    finally:
        restore()


def _turn(turn_id, text, segments=()):
    return {"turn_id": turn_id, "role": "user", "origin": "typed", "context": [], "text": text,
            "segments": [list(s) for s in segments]}


def test_ar11_a_ranker_that_fails_is_none(caplog, monkeypatch):
    """AR11: a ranker raising under the lock: the share ranks, the window and the block are composed, said."""
    loaded, restore = oj.open_window()
    lib, peels, composer = loaded[oj.LIBRARIAN], loaded[oj.PEELS], loaded[oj.COMPOSER]
    try:
        cfg = oj.config(lib)
        units = list(_COMMON * 6) + [_RARE]
        budget = composer.Budget(window=780, reserve=60, core=60, receipts=300, peels=peels.estimate_tokens(_RARE),
                                 flesh=140, turn=60)

        def broken(query, texts, contexts):
            raise sqlite3.OperationalError("no such tokenizer: porter")

        monkeypatch.setattr(peels, "anchor_bm25", broken)
        _held(lib, "c1", units, cfg)
        with caplog.at_level(logging.WARNING):
            block = str(lib.memory_block("c1", _QUESTION, budget=budget))
        assert block and _RARE not in block and any(c in block for c in _COMMON), ("the share ranks", block)
        _state, messages = _held(lib, "c2", units, cfg)
        win = oj.window(loaded, "c2", _QUESTION, messages=messages, config=cfg, model_window=4096,
                        budgets=oj.snug(loaded, _QUESTION))
        assert win.refusal is None, ("the window is composed", win.refusal)
        assert any("anchor rank" in r.getMessage() and "OperationalError" in r.getMessage() for r in caplog.records), (
            [r.getMessage() for r in caplog.records])
    finally:
        restore()


def test_ar12_the_anchors_own_words_break_a_tie():
    """AR12: a ranker scoring both anchors alike: the one holding more of the question first, as by the share."""
    loaded, restore = oj.open_window()
    peels, receipts = loaded[oj.PEELS], loaded[_RECEIPTS]
    try:
        cat, rex = "My cat likes my dog.", "My dog Rex is a beagle."
        said = cat + " " + rex
        span = [_turn("u1", said)]
        cellar, ledger, _key = _receipt(receipts, span, (("u1", 0, len(cat)), ("u1", len(cat) + 1, len(said))))
        query = "Which breed is my dog Rex?"
        cap = max(peels.estimate_tokens(cat), peels.estimate_tokens(rex))
        assert cap < peels.estimate_tokens(cat) + peels.estimate_tokens(rex), "control: the cap keeps one"
        def alike(query, texts, contexts):
            return [1.0] * len(texts)

        shown = peels.select_anchors(ledger, cellar, query, cap, rank=alike)
        assert [a.text for a in shown] == [rex], ("the one whose words hold more of the question", shown)
    finally:
        restore()


def _laid(peels, ledger, cellar, query, texts):
    """The first anchor a cap that keeps one of ``texts`` lays, ranked by BM25."""
    cap = max(peels.estimate_tokens(t) for t in texts)
    return [a.text for a in peels.select_anchors(ledger, cellar, query, cap, rank=peels.anchor_bm25)][:1]


def test_ar13_the_questions_terms_not_its_function_words():
    """AR13: function words never rank ("what", "did", "about"); a two-letter name does: the answer is laid, as
    by the share."""
    loaded, restore = oj.open_window()
    peels, receipts = loaded[oj.PEELS], loaded[_RECEIPTS]
    try:
        cases = (("What did Caroline say about the adoption agency?",
                  ["Caroline chose the Hope adoption agency.", "Caroline did not care about what we did.",
                   "Caroline likes tea.", "Caroline runs daily.", "Caroline reads books.", "Caroline sings well."]),
                 ("When did Alice adopt the dog?", ["Alice adopted a dog.", "The dog did that when the storm came."]),
                 ("Which town does Jo live in?", ["Jo lives in Lyon now.", "Where I live does not matter."]),
                 ("What did the AI say about the UK trip?",
                  ["The AI said the UK trip is off.",
                   "What did you say about the trip and the hotel?  The trip was fine."]))
        for query, texts in cases:
            span = [_turn(f"u{n}", t) for n, t in enumerate(texts, 1)]
            cellar, ledger, _key = _receipt(receipts, span, _whole(span))
            assert _laid(peels, ledger, cellar, query, texts[:2]) == [texts[0]], (query, texts[0])
    finally:
        restore()


def test_ar14_a_text_counted_once():
    """AR14: the rare sentence kept by three receipts is one row: it still ranks first, and is laid."""
    loaded, restore = oj.open_window()
    peels, receipts = loaded[oj.PEELS], loaded[_RECEIPTS]
    try:
        others = ["Caroline met Melanie in the park.", "Melanie and Caroline like the park.",
                  "Caroline told Melanie about the park."]
        spans = [_alone(f"r{n}", _RARE) for n in range(3)] + [_alone(f"o{n}", t) for n, t in enumerate(others)]
        cellar, ledger = _receipts_of(receipts, spans)
        ranked = peels.select_anchors(ledger, cellar, _QUESTION, 10 ** 6, rank=peels.anchor_bm25)
        assert ranked[0].text == _RARE, [a.text for a in ranked]
        assert _laid(peels, ledger, cellar, _QUESTION, [_RARE] + others) == [_RARE]
    finally:
        restore()


def test_ar15_the_newest_first_among_equals():
    """AR15: a fact and its update hold the same terms of the question: the newer receipt's is laid, longer or not."""
    loaded, restore = oj.open_window()
    peels, receipts = loaded[oj.PEELS], loaded[_RECEIPTS]
    try:
        for old, new, filler in (("Alice lives in Berlin.", "Alice now lives in Oslo.", 0),
                                 ("Alice lives in Berlin.", "Alice now lives in Oslo.", 6),
                                 ("Alice lives in Berlin.", "Alice lives in Oslo since May.", 6)):
            cellar, ledger = receipts.Cellar(), receipts.ReceiptLedger()
            spans = [[_turn("u1", old)]] + [[_turn(f"f{n}", f"Alice met Bob on day {n}.")] for n in range(filler)]
            for span in spans + [[_turn("u9", new)]]:
                key = cellar.store(span)
                ledger.append(receipts.make_receipt(span, key, "held", _whole(span)))
            assert _laid(peels, ledger, cellar, "Where does Alice live?", [old, new]) == [new], (old, new, filler)
    finally:
        restore()


def _receipts_of(receipts, spans):
    """A ledger of one held receipt per ``(turns, anchors)``, oldest first."""
    cellar, ledger = receipts.Cellar(), receipts.ReceiptLedger()
    for span, anchors in spans:
        key = cellar.store(span)
        ledger.append(receipts.make_receipt(span, key, "held", anchors))
    return cellar, ledger


def _alone(turn_id, text):
    span = [_turn(turn_id, text)]
    return span, _whole(span)


def test_ar18_equals_never_pass_a_segment():
    """AR18: a newer "Alice in Oslo." does not put the segment's fact before the segment, nor show it twice."""
    loaded, restore = oj.open_window()
    peels, receipts = loaded[oj.PEELS], loaded[_RECEIPTS]
    try:
        other = "Alice in Oslo."
        held = ([_turn("u1", _BESIDE)], (("u1", 0, len(_BESIDE)), ("u1", 0, len(_FACT))))
        for spans in ([held], [held, _alone("v1", other)]):
            cellar, ledger = _receipts_of(receipts, spans)
            ranked = [a.text for a in peels.select_anchors(ledger, cellar, "alice in oslo and the logs?", 10 ** 6,
                                                           rank=peels.anchor_bm25)]
            assert _FACT not in ranked and any(_ORDER in text for text in ranked), ranked
    finally:
        restore()


def test_ar19_a_term_counts_by_its_stem():
    """AR19: "Where does Alice live?": the update, then the stale fact, then the anchors naming Alice alone."""
    loaded, restore = oj.open_window()
    peels, receipts = loaded[oj.PEELS], loaded[_RECEIPTS]
    try:
        old, new = "Alice lives in Berlin.", "Alice now lives in Oslo."
        fillers = [f"Alice met Bob on day {n}." for n in range(3)]
        spans = [_alone("u1", old), _alone("u2", new)] + [_alone(f"f{n}", t) for n, t in enumerate(fillers)]
        cellar, ledger = _receipts_of(receipts, spans)
        ranked = [a.text for a in peels.select_anchors(ledger, cellar, "Where does Alice live?", 10 ** 6,
                                                       rank=peels.anchor_bm25)]
        assert ranked[:2] == [new, old], ranked
        assert _laid(peels, ledger, cellar, "Where does Alice live?", [old, new] + fillers) == [new]
    finally:
        restore()


def test_ar20_any_script_folded_alike():
    """AR20: "what did Maik buy?" in Russian: the anchor naming him is laid, as the share lays it."""
    loaded, restore = oj.open_window()
    peels, receipts = loaded[oj.PEELS], loaded[_RECEIPTS]
    try:
        for name in ("\u041c\u0430\u0439\u043a", "Mike"):
            texts = [f"{name} \u043a\u0443\u043f\u0438\u043b \u0432\u0435\u043b\u043e\u0441\u0438\u043f\u0435\u0434.",
                     "\u042f \u043a\u0443\u043f\u0438\u043b \u0445\u043b\u0435\u0431.",
                     "\u041a\u0443\u043f\u0438\u043b \u043c\u043e\u043b\u043e\u043a\u043e."]
            query = f"\u0427\u0442\u043e \u043a\u0443\u043f\u0438\u043b {name}?"
            span = [_turn(f"u{n}", t) for n, t in enumerate(texts, 1)]
            cellar, ledger, _key = _receipt(receipts, span, _whole(span))
            share = peels.select_anchors(ledger, cellar, query, max(peels.estimate_tokens(t) for t in texts))
            assert [a.text for a in share][:1] == [texts[0]], ("control: the share lays it", name)
            assert _laid(peels, ledger, cellar, query, texts) == [texts[0]], name
            scores = peels.anchor_bm25(query, texts)
            assert scores[0] > max(scores[1:]), ("the name counts in BM25", name, scores)
    finally:
        restore()


def test_ar16_equals_are_the_anchors_that_hold_the_same_terms():
    """AR16: anchors holding the rare term all rank before those holding only the common one; one of them is laid."""
    loaded, restore = oj.open_window()
    peels, receipts = loaded[oj.PEELS], loaded[_RECEIPTS]
    try:
        adopted, glad = "Caroline adopted a greyhound.", "Melanie was glad."
        said = adopted + " " + glad
        radio = "Greyhound racing is cruel, said the vet on the radio this morning in a long talk."
        fillers = ["Melanie likes tea.", "Melanie runs daily.", "Melanie reads books.", "Melanie sings well."]
        spans = [([_turn("u1", said)], (("u1", 0, len(adopted)), ("u1", len(adopted) + 1, len(said)))),
                 _alone("r1", radio)] + [_alone(f"f{n}", t) for n, t in enumerate(fillers)]
        cellar, ledger = _receipts_of(receipts, spans)
        query = "Where is the greyhound of Melanie?"
        ranked = [a.text for a in peels.select_anchors(ledger, cellar, query, 10 ** 6, rank=peels.anchor_bm25)]
        rare, common = [adopted, radio], [glad] + fillers
        assert max(ranked.index(t) for t in rare) < min(ranked.index(t) for t in common), ranked
        assert _laid(peels, ledger, cellar, query, [adopted] + fillers)[0] in rare
    finally:
        restore()


def test_ar17_an_anchors_own_words_only():
    """AR17: "Alice met Carol at noon." in a newer turn that says "I live near the station" never passes the update."""
    loaded, restore = oj.open_window()
    peels, receipts = loaded[oj.PEELS], loaded[_RECEIPTS]
    try:
        old, new = "Alice lives in Berlin.", "Alice now lives in Oslo."
        met, near = "Alice met Carol at noon.", "I live near the station."
        said = met + " " + near
        spans = [_alone("u1", old)] + [_alone(f"f{n}", f"Alice met Bob on day {n}.") for n in range(4)]
        spans += [_alone("u2", new), ([_turn("u3", said)], (("u3", 0, len(met)),))]
        cellar, ledger = _receipts_of(receipts, spans)
        ranked = [a.text for a in peels.select_anchors(ledger, cellar, "Where does Alice live?", 10 ** 6,
                                                       rank=peels.anchor_bm25)]
        assert ranked[:2] == [new, old] and ranked.index(met) > 1, ranked
        assert _laid(peels, ledger, cellar, "Where does Alice live?", [old, new, met]) == [new]
    finally:
        restore()


_MOVED = ("Alice moved to Oslo last spring with her two children, the old dog from the farm, three boxes of books, a "
          "piano she never plays and the bicycle her brother gave her for her fortieth birthday.")


def test_ar21_below_the_floor_the_terms_held_decide():
    """AR21: two anchors, every weight floored: the long one holding both terms is laid, in either order; four
    anchors: the long update holding "lives" is laid before short anchors that only name Alice."""
    loaded, restore = oj.open_window()
    peels, receipts = loaded[oj.PEELS], loaded[_RECEIPTS]
    try:
        short = "Alice laughed."
        for order in ((_MOVED, short), (short, _MOVED)):
            cellar, ledger = _receipts_of(receipts, [_alone(f"u{n}", t) for n, t in enumerate(order)])
            assert _laid(peels, ledger, cellar, "Is Alice in Oslo?", list(order)) == [_MOVED], order
        old = ("Alice lives in Berlin, in the flat on the long canal street near the old harbour market, with two "
               "cats, a piano and the noisy neighbours upstairs who play drums at night.")
        new = ("Alice lives in Oslo now, in the flat on the long fjord street near the new harbour market, with two "
               "cats, a piano and quiet neighbours upstairs who never play anything.")
        spans = [_alone("u1", old), _alone("f0", "Alice smiled."), _alone("f1", "Alice sang."), _alone("u2", new)]
        cellar, ledger = _receipts_of(receipts, spans)
        cap = peels.estimate_tokens(new)
        laid = [a.text for a in peels.select_anchors(ledger, cellar, "Where does Alice live?", cap,
                                                     rank=peels.anchor_bm25)]
        assert laid == [new], laid
    finally:
        restore()


def test_ar22_a_fact_inside_its_segment_counted_once():
    """AR22: the fact kept alone and within its segment: the segment, holding the rare term, is laid first."""
    loaded, restore = oj.open_window()
    peels, receipts = loaded[oj.PEELS], loaded[_RECEIPTS]
    try:
        fact = "Caroline adopted a greyhound."
        beside = fact + " " + _ORDER
        fillers = ["Melanie likes tea.", "Melanie runs daily."]
        spans = [([_turn("u1", beside)], (("u1", 0, len(beside)), ("u1", 0, len(fact))))]
        spans += [_alone(f"f{n}", t) for n, t in enumerate(fillers)]
        cellar, ledger = _receipts_of(receipts, spans)
        ranked = [a.text for a in peels.select_anchors(ledger, cellar, "Where is the greyhound of Melanie?", 10 ** 6,
                                                       rank=peels.anchor_bm25)]
        assert ranked[0] == beside and fact not in ranked, ranked
    finally:
        restore()
