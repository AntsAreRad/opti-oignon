#!/usr/bin/env python3
"""Contracts for the engine without backtracking: a pattern the linear check refuses, read as ``re`` reads it, in
time linear in the text.

``linear_engine`` compiles a pattern into a Pike program and reads it with
two passes: one from the end of the text that finds where a match can still
be completed, and a Pike machine from each leftmost start that drops every
thread which cannot, so that each match reads its own span only. What it
answers is held to ``re`` itself, on a seeded corpus and on tables of the
rules where a machine without backtracking most easily departs from it (empty
rounds, groups in lookarounds, possessive repeats, anchors around characters
beyond ASCII). What it costs is held by its own count of work, never by a
clock: doubling the text doubles the work.

  * LE1 -- every call a caller makes answers as ``re`` answers on a seeded
    corpus of generated patterns: ``search``, ``match``, ``fullmatch``,
    ``finditer``, ``findall``, ``split``, ``sub`` and ``subn``, with the spans
    of every group, ``lastindex``, ``lastgroup``, ``pos`` and ``endpos``.
  * LE2 -- empty rounds: a loop whose round can read nothing takes it once,
    then leaves, as ``re`` does, greedy, lazy and counted, nested too.
  * LE3 -- a pattern exponential for ``re`` is read in linear work, by a
    search and by every match in turn.
  * LE4 -- every match in turn is read in linear work where a higher branch
    runs to the end of the text after each match.
  * LE5 -- lookarounds are read in linear work, ahead and behind.
  * LE6 -- the groups of a positive lookaround, nested ones too, are those
    ``re`` gives, and the last group closed through them; one a repeat may
    pass twice, or a lookahead not bounded in width, is refused by name.
  * LE7 -- a possessive repeat or an atomic group of one set of characters is
    read exactly; of more, refused by name.
  * LE8 -- a back-reference, a conditional and a bytes pattern are refused by
    name, each a ``re.error``.
  * LE9 -- what a pattern may cost the engine is bounded: instructions, sets
    of characters, passes over every code point, each refused by name and at
    once.
  * LE10 -- ``compile_linear`` gives ``re`` a pattern the check calls linear,
    the engine one it refuses, and refuses with both reasons one neither
    reads; with the engine left out, or where it cannot be loaded, it refuses
    as before.
  * LE11 -- ``sub`` and ``subn`` expand a template as ``re`` does, and take a
    function and a count as it does; an invalid template raises its error.
  * LE12 -- the match and the pattern answer as ``re``'s do for every call
    and attribute a caller reads; a compiled pattern is a value, as ``re``'s
    is: copied as itself, pickled as the call that makes it.
  * LE13 -- the windows of the backward pass change no answer, and a window
    is read again when the machine reaches it.
  * LE14 -- a cache bounded to one, two or three moves changes no answer,
    and no cache holds more moves than its bound.
  * LE15 -- the work per character of a text is bounded by the size of the
    program, and does not grow with it where the states the text visits
    repeat (a list of words on a text with no match).
  * LE16 -- the prefilter is sound: a text that holds none of its strings
    holds no match, and every match holds one of them.
  * LE17 -- the backward pass keeps one set every WINDOW positions and one
    window, whatever the length of the text.
  * LE18 -- a boundary in an empty text follows the running interpreter, and
    a version the engine was not measured on is refused by name.
  * LE19 -- one read of a pattern at a time: a second read waits while the
    first builds a move, and every move kept is the one its key names.
  * LE20 -- characters beyond ASCII: word characters, case folding,
    boundaries and line ends answer as ``re``'s do.
  * LE21 -- ``match`` with ``pos`` past ``endpos``, where the engine still
    tries the pattern and every read fails: each anchor, lookaround and
    repeat answers as ``re``'s does there.
  * LE22 -- a position holds at most m * (d + 1) threads of the Pike machine
    (m instructions, d the depth of nullable loops), copies for the empty
    rounds included.
  * LE23 -- the cases a pass of mutants found the other contracts blind to
    (a jump or a split of a lookbehind's forward pass, the empty rounds of a
    counted repeat, a lookahead that never holds, the anchors past the end):
    each answers as ``re`` answers it.
  * LE24 -- a search tries a start where re's search does: in the set re
    compiles from the pattern's first class under the pattern's own type
    flags, even where a group of other type flags makes re's search skip a
    start its match takes.
  * LE25 -- a reader that takes one match after another from a moving
    position (``match`` or ``search`` with ``pos``) reads the text in linear
    work: the passes of a text are kept for its next read.
  * LE26 -- a subclass of ``str`` is read as its exact text: none of its
    methods runs during a read, and the match names the object given.
  * LE27 -- ``finditer`` is lazy, as re's: a step finds one match, and other
    reads of the pattern between two steps change nothing.
  * LE28 -- the cases the last pass of mutants found the other contracts
    blind to (the key that merges two threads at one instruction, the groups
    of a lookahead left unset, a lookbehind's forward pass, a lookbehind whose
    rounds read nothing): each answers as ``re`` answers it; and a refusal
    carries what a ``re.error`` carries.
  * LE29 -- the shapes a generator of patterns long missed (groups whose
    rounds may read nothing, with a count; nested lookarounds, counted groups,
    atomic and possessive parts in a lookbehind's body; a match past endpos)
    answer as ``re`` answers them, attributes included, on a seeded corpus.
"""
import copy
import pickle
import random
import re
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolation import isolate, source  # noqa: E402

# The versions of Python the engine was measured on against re (LE18 holds this to the engine's own list): on any
# other the engine refuses every pattern, compile_linear refuses as before, and these contracts are skipped by name.
_MEASURED = ((3, 11), (3, 12), (3, 13), (3, 14))
pytestmark = pytest.mark.skipif(sys.version_info[:2] not in _MEASURED,
                                reason="the engine without backtracking is not measured on this version of Python")

BUDGET_S = {
    "test_le1_every_call_answers_as_re_on_a_seeded_corpus": 30.0,
    "test_le2_an_empty_round_is_taken_once_then_the_loop_is_left_as_re_does": 6.0,
    "test_le3_a_pattern_exponential_for_re_is_read_in_linear_work": 10.0,
    "test_le4_every_match_in_turn_is_read_in_linear_work_where_a_higher_branch_runs_to_the_end": 10.0,
    "test_le5_lookarounds_are_read_in_linear_work": 10.0,
    "test_le6_the_groups_of_a_positive_lookaround_are_those_re_gives": 6.0,
    "test_le7_a_possessive_repeat_or_an_atomic_group_of_one_set_is_read_exactly": 6.0,
    "test_le8_a_back_reference_a_conditional_and_a_bytes_pattern_are_refused_by_name": 6.0,
    "test_le9_what_a_pattern_may_cost_the_engine_is_bounded_and_refused_at_once": 10.0,
    "test_le10_compile_linear_gives_re_the_linear_the_engine_the_rest_and_refuses_what_neither_reads": 10.0,
    "test_le11_a_template_a_function_and_a_count_are_taken_as_re_takes_them": 6.0,
    "test_le12_the_match_and_the_pattern_answer_as_re_s_do": 6.0,
    "test_le13_the_windows_of_the_backward_pass_change_no_answer": 20.0,
    "test_le14_a_cache_of_one_to_three_moves_changes_no_answer_and_none_holds_more_than_its_bound": 20.0,
    "test_le15_the_work_per_character_is_bounded_by_the_program_and_does_not_grow_where_its_states_repeat": 20.0,
    "test_le16_the_prefilter_is_sound": 20.0,
    "test_le17_the_backward_pass_keeps_one_set_a_window_whatever_the_length": 10.0,
    "test_le18_a_boundary_in_an_empty_text_follows_the_interpreter_and_an_unmeasured_version_is_refused": 6.0,
    "test_le19_one_read_of_a_pattern_at_a_time": 10.0,
    "test_le20_characters_beyond_ascii_answer_as_re_s_do": 6.0,
    "test_le21_a_match_past_endpos_answers_as_re_s_does": 6.0,
    "test_le22_a_position_holds_at_most_m_times_d_plus_one_threads": 6.0,
    "test_le23_the_cases_a_pass_of_mutants_found_unheld_answer_as_re_does": 6.0,
    "test_le24_a_search_tries_a_start_where_re_s_search_does": 6.0,
    "test_le25_a_reader_that_takes_one_match_after_another_reads_the_text_in_linear_work": 20.0,
    "test_le26_a_subclass_of_str_is_read_as_its_exact_text_and_runs_no_code_during_a_read": 6.0,
    "test_le27_finditer_is_lazy_and_other_reads_between_two_steps_change_nothing": 10.0,
    "test_le28_the_cases_the_last_pass_of_mutants_found_unheld_answer_as_re_does": 6.0,
    "test_le29_the_shapes_a_generator_long_missed_answer_as_re_does": 20.0,
}


def _loaded():
    """(linear_pattern, linear_engine, restore): both loaded from their files in one window."""
    loaded, restore = isolate(targets={"opti_oignon.linear_pattern": source("linear_pattern.py"),
                                       "opti_oignon.linear_engine": source("linear_engine.py")})
    return loaded["opti_oignon.linear_pattern"], loaded["opti_oignon.linear_engine"], restore


def _regs(found):
    return None if found is None else (found.regs, found.lastindex, found.lastgroup, found.pos, found.endpos)


def _answers(rx, text, pos=0, endpos=None, whole=True):
    """Everything a caller can read of a pattern on one text, for ``re`` and the engine alike."""
    args = (text, pos) if endpos is None else (text, pos, endpos)
    out = [_regs(rx.search(*args)), [_regs(found) for found in rx.finditer(*args)], _regs(rx.match(*args)),
           _regs(rx.fullmatch(*args)), rx.findall(*args)]
    if whole:
        template = r"[\g<0>|\1]" if rx.groups else r"[\g<0>]"
        out += [rx.split(text), rx.split(text, 1), rx.sub(template, text), rx.subn(template, text, 1)]
    return out


# A generator of patterns: classes, anchors, groups (some named), alternatives, every kind of repeat, groups of scoped
# flags, lookarounds whose bodies are generated alike (ahead and behind, positive and negative, nested), possessive
# and atomic forms. Wide (LE29), it also makes the shapes it long missed: groups whose rounds may read nothing, with a
# count; nested lookarounds, counted groups, atomic and possessive parts in a lookbehind's body.
_ATOMS = ["a", "b", "x", " ", r"\s", r"\w", r"\d", r"\W", ".", "[ab]", "[^a]", "[a-z]", r"\n", "-", "k", "",
          r"\b", r"\B", "^", "$", r"\A", r"\Z", "(?>a*)", "(?>[ab]{1,2}?)", "(?>x)"]
_FIXED = ["a", "b", "x", " ", r"\w", r"\W", ".", "[ab]", "[^a]", r"\n", "", r"\b", "^", "$"]
_COUNTS = ["", "", "", "*", "+", "?", "{2}", "{1,3}", "{2,}", "*?", "+?", "??", "{0,2}?", "{0,3}", "{1,2}?",
           "*+", "++", "?+", "{1,3}+"]
_FLAGS = [0, re.I, re.M, re.S, re.A, re.I | re.M, re.M | re.S]
_ZERO = ("", r"\b", r"\B", "^", "$", r"\A", r"\Z")
_ALPHABET = "abx -k1\nA_"


def _gen(rng, depth, state, wide=False):
    out = []
    for _ in range(rng.randint(1, 4)):
        roll = rng.random()
        if roll < 0.12 and depth < 3:
            out.append("(?:" + _gen(rng, depth + 1, state, wide) + "|" + _gen(rng, depth + 1, state, wide) + ")"
                       + rng.choice(_COUNTS))
        elif roll < 0.22 and depth < 3:
            state["groups"] += 1
            name = f"?P<g{state['groups']}>" if rng.random() < 0.2 else ""
            out.append("(" + name + _gen(rng, depth + 1, state, wide) + ")" + rng.choice(_COUNTS))
        elif roll < 0.27 and depth < 3:
            scoped = rng.choice(["?i:", "?-i:", "?a:", "?s:", "?m:", "?is:", "?i-s:", "?u:", "?x:"])
            out.append("(" + scoped + _gen(rng, depth + 1, state, wide) + ")" + rng.choice(_COUNTS))
        elif roll < 0.34 and depth < 3:
            kind = rng.choice(["?=", "?!", "?<=", "?<!"])
            body = _fixed(rng, depth + 1, state, wide) if kind.startswith("?<") and rng.random() < 0.7 else \
                _gen(rng, depth + 1, state, wide)
            out.append("(" + kind + body + ")")
        elif wide and roll < 0.40 and depth < 3:
            out.append(_nothing(rng, depth + 1, state) + rng.choice(_COUNTS[3:]))
        else:
            atom = rng.choice(_ATOMS)
            zero = atom in _ZERO or atom.startswith("(?")
            out.append(atom + ("" if zero else rng.choice(_COUNTS)))
    return "".join(out)


def _fixed(rng, depth, state, wide=False):
    out = []
    for _ in range(rng.randint(1, 3)):
        roll = rng.random()
        if roll < 0.25 and depth < 3:
            state["groups"] += 1
            out.append("(" + _fixed(rng, depth + 1, state, wide) + ")")
        elif roll < 0.35:
            out.append("(?:" + rng.choice(["a|b", "ab|xk", r"\w|-", "a|"]) + ")")
        elif wide and roll < 0.45 and depth < 3:
            out.append("(?:" + rng.choice(list(_ZERO) + ["(?=a)", "(?!b)", r"(?<=\w)", "(?<!a)"]) + ")"
                       + rng.choice(["*", "+", "?", "{2}", "{0,3}", "*?", "+?", "{1,2}?"]))
        elif wide and roll < 0.52 and depth < 3:
            kind = rng.choice(["?=", "?!", "?<=", "?<!"])
            body = _fixed(rng, depth + 1, state, wide) if kind.startswith("?<") else _gen(rng, depth + 1, state, wide)
            out.append("(" + kind + body + ")")
        elif wide and roll < 0.60 and depth < 3:
            out.append("(?:" + _fixed(rng, depth + 1, state, wide) + ")" + rng.choice(["{2}", "{1}", "{0}", "{2}+", "{2}?"]))
        elif wide and roll < 0.66:
            out.append(rng.choice(["(?>ab)", "(?>a|b)", "[ab]{2}+", "(?>x)", r"(?>\w)", "a{1}+"]))
        else:
            atom = rng.choice(_FIXED)
            out.append(atom + (rng.choice(["", "", "{2}"]) if atom not in _ZERO else ""))
    return "".join(out)


def _nothing(rng, depth, state):
    """A group whose rounds may read nothing: zero-width atoms, empty alternatives, lookarounds, optional parts."""
    parts = []
    for _ in range(rng.randint(1, 2)):
        roll = rng.random()
        if roll < 0.5:
            parts.append(rng.choice(_ZERO))
        elif roll < 0.7 and depth < 3:
            parts.append("(?" + rng.choice(["=", "!", "<=", "<!"]) + _fixed(rng, depth + 1, state, True) + ")")
        else:
            parts.append(rng.choice(["a?", "[ab]*", "x??", "(?:a|)", r"\w{0,1}"]))
    body = rng.choice(["", "|"]).join(parts) if len(parts) > 1 else parts[0]
    if rng.random() < 0.4:
        state["groups"] += 1
        return "(" + body + ")"
    return "(?:" + body + ")"


def _corpus(engine, seed, patterns, texts, longest, wide=False):
    """[(pattern, flags, compiled, read, [(text, pos, endpos)])]: generated patterns ``re`` compiles and the engine
    reads, each with its texts; and the count of those the engine refused. Wide: the shapes of ``_gen`` it long
    missed, characters beyond ASCII, and a pos past the end or an endpos before pos as often as not."""
    rng, out, refused, drawn = random.Random(seed), [], 0, 0
    alphabet = _ALPHABET + "\x85\xe9\u0660\u2028\u212a\xdf" if wide else _ALPHABET
    while len(out) < patterns:
        drawn += 1
        assert drawn <= 4 * patterns, f"the engine reads {len(out)} of {drawn} generated patterns"
        pattern, flags = _gen(rng, 0, {"groups": 0}, wide), rng.choice(_FLAGS)
        try:
            compiled = re.compile(pattern, flags)
        except (re.error, ValueError, OverflowError, RecursionError):
            continue
        try:
            read = engine.compile(pattern, flags)
        except engine.Unreadable:
            refused += 1
            continue
        cases = []
        for _ in range(texts):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, longest)))
            if wide:
                pos = rng.choice([0, 0, 0, 1, 2, len(text), len(text) + 1])
                endpos = rng.choice([None, None, None, max(0, len(text) - 1), len(text) + 3, max(0, pos - 1)])
                cases.append((text, pos, endpos))
                continue
            cases.append((text, rng.choice([0, 0, 0, 1, 2]),
                          rng.choice([None, None, None, max(0, len(text) - 1), len(text) + 3])))
        out.append((pattern, flags, compiled, read, cases))
    return out, refused


def _differ(corpus, whole_every=3):
    """(checks, divergences, kinds): the corpus read by ``re`` and the engine; kinds counts what the answers held."""
    checks, divergences, kinds = 0, [], dict(matched=0, groups=0, last=0, several=0, empty=0)
    for pattern, flags, compiled, read, cases in corpus:
        for number, (text, pos, endpos) in enumerate(cases):
            whole = number % whole_every == 0
            want = _answers(compiled, text, pos, endpos, whole)
            got = _answers(read, text, pos, endpos, whole)
            checks += 1
            if got != want and len(divergences) < 5:
                divergences.append((pattern, flags, text, pos, endpos))
            found = want[0]
            if found is not None:
                kinds["matched"] += 1
                kinds["groups"] += any(span != (-1, -1) for span in found[0][1:])
                kinds["last"] += found[1] is not None
                kinds["empty"] += found[0][0][0] == found[0][0][1]
            kinds["several"] += len(want[1]) > 1
    return checks, divergences, kinds


# ---------------------------------------------------------------------------
# LE1 -- every call answers as re on a seeded corpus
# ---------------------------------------------------------------------------
def test_le1_every_call_answers_as_re_on_a_seeded_corpus():
    _lp, le, restore = _loaded()
    try:
        corpus, refused = _corpus(le, 1401, 900, 6, 7)
        checks, divergences, kinds = _differ(corpus)
        assert not divergences, divergences
        assert checks == 900 * 6, checks
        # Not blind: the corpus reaches what the rules are about, and the answers hold matches of every kind.
        held = [pattern for pattern, *_ in corpus]
        assert sum("(?=" in p or "(?!" in p or "(?<" in p for p in held) > 100, "lookarounds"
        assert sum(re.search(r"\((?:\?P<\w+>)?[^?]", p) is not None and "(?=" in p for p in held) > 20, "groups"
        assert sum("+?" in p or "*?" in p or "??" in p for p in held) > 100, "lazy repeats"
        assert sum("++" in p or "*+" in p or "?+" in p or "(?>" in p for p in held) > 50, "possessive and atomic"
        assert sum(re.search(r"\(\?-?[aimsux]+(?:-[aimsux]+)?:", p) is not None for p in held) > 50, "scoped flags"
        assert all(count > 50 for count in kinds.values()), kinds
        assert refused > 0, "the generator reaches patterns the engine refuses"
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE2 -- empty rounds
# ---------------------------------------------------------------------------
_LE2_TABLE = [
    (r"(a|)*b", ["aab", "b", "ab", "ac"]), (r"(?:(a)|b)*", ["ab", "ba", "aab"]), (r"(a|)*", ["aa", "", "b"]),
    (r"(a|)*?b", ["aab", "b"]), (r"(a?)*?b", ["aab"]), (r"(a|)+", ["aa", ""]), (r"(a|){2,5}", ["a", "aaaaaa", ""]),
    (r"(a|){2,5}?", ["aaa", ""]), (r"((a|)*)*", ["aa", "ab"]), (r"((a)|(b)|)*c", ["abac", "c"]),
    (r"(?:(?:|){3}a)?", ["a", "aa", ""]), (r"(?:x*)*y", ["xxy", "y"]), (r"((?:a|)*?)*b", ["aab"]),
    (r"(\b|a)*", ["aa", " a"]), (r"(?:(a)|)*?(?:(b)|)*", ["abab", "ba"]), (r"(a*)+$", ["aaa", ""]),
    (r"(?:(a)|(?=b)|c)*", ["acb", "b"]), (r"((a)|(?:))+?x", ["aax", "x"]),
]


def test_le2_an_empty_round_is_taken_once_then_the_loop_is_left_as_re_does():
    _lp, le, restore = _loaded()
    try:
        for pattern, texts in _LE2_TABLE:
            compiled, read = re.compile(pattern), le.compile(pattern)
            for text in texts:
                assert _answers(read, text) == _answers(compiled, text), (pattern, text)
        # Not blind: re takes an empty round once, after one that read (the group ends empty at the end), and a
        # loop of empty rounds alone ends at once.
        assert re.match(r"(a|)*", "aa").span(1) == (2, 2) and le.compile(r"(a|)*").match("aa").span(1) == (2, 2)
        assert re.match(r"(?:(a)|b)*", "ab").span(1) == (0, 1), "a group not set in the last round keeps its value"
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE3 -- a pattern exponential for re is read in linear work
# ---------------------------------------------------------------------------
def _works(read, call, make, sizes=(2000, 4000, 8000)):
    """The engine's work for ``call`` on texts of each size, and the answers' lengths."""
    works, sizes_out = [], []
    for size in sizes:
        read.work = 0
        answer = getattr(read, call)(make(size))
        answer = list(answer) if call == "finditer" else answer
        works.append(read.work)
        sizes_out.append(len(answer) if isinstance(answer, (list, str)) else (answer is not None))
    return works, sizes_out


def _linear(works):
    """True when each doubling of the text at most doubles the work, give or take a constant part."""
    return all(0 < small and large <= 2.25 * small for small, large in zip(works, works[1:]))


# Each with a text it reads with no match, a text where it finds one match a few characters long after another, and
# a short one it matches from its start, where re finds the match at once (on any text where a start fails, re takes
# exponential time).
_LE3_EXPONENTIAL = [(r"(?:a|a)*b", lambda n: "a" * n + "!b", lambda n: "ab" * n, "aab"),
                    (r"(a+)+b", lambda n: "a" * n + "!b", lambda n: "aab" * n, "aab"),
                    (r"(\w+\s?)*$", lambda n: "a" * n + "!", None, "a a"),
                    (r"(?:(?:|){40}a)?b", lambda n: "a" * n + "!b", lambda n: "ab" * n, "ab"),
                    (r"(x+x+)+y", lambda n: "x" * n + "!y", lambda n: "xxy" * n, "xxy")]


def test_le3_a_pattern_exponential_for_re_is_read_in_linear_work():
    lp, le, restore = _loaded()
    try:
        for pattern, make, many, short in _LE3_EXPONENTIAL:
            assert lp.verdict(pattern) == "exponential", ("the witness is exponential for re", pattern)
            read = le.compile(pattern)
            for call in ("search", "finditer"):
                works, _sizes = _works(read, call, make, (1000, 2000, 4000))
                assert _linear(works), (pattern, call, works)
            if many is not None:
                works, sizes = _works(read, "finditer", many, (1000, 2000, 4000))
                assert _linear(works) and sizes == [1000, 2000, 4000], (pattern, "every match", works, sizes)
            found = re.compile(pattern).search(short)
            assert found is not None and _regs(read.search(short)) == _regs(found), ("as re on a short text", pattern)
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE4 -- every match in turn in linear work where a higher branch runs to the end
# ---------------------------------------------------------------------------
_LE4_PATTERNS = [(r"b.*c|b", "b"), (r"x\w*z|x", "x"), (r"(?:a[^\n]*\n|a)", "a"), (r"(?:a(?=\w*q)|a)", "a")]


def test_le4_every_match_in_turn_is_read_in_linear_work_where_a_higher_branch_runs_to_the_end():
    _lp, le, restore = _loaded()
    try:
        for pattern, unit in _LE4_PATTERNS:
            read = le.compile(pattern)
            for call in ("finditer", "findall", "split"):
                works, sizes = _works(read, call, lambda n, unit=unit: unit * n)
                assert _linear(works), (pattern, call, works)
                assert sizes[0] >= 2000 and sizes[2] >= 8000, ("one match a character", pattern, call, sizes)
            read.work = 0
            text = unit * 4000
            assert read.sub("-", text) == re.sub(pattern, "-", unit * 4000), pattern
            # Not blind: re itself reads every match from its start to the end of the text.
            assert re.compile(pattern).findall(unit * 300) == read.findall(unit * 300), pattern
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE5 -- lookarounds in linear work
# ---------------------------------------------------------------------------
_LE5_PATTERNS = [(r"(?=.*x)\w", "a"), (r"\w(?!\w*z)", "a"), (r"(?<=a)b", "ab"), (r"(?<!x)a(?=a*\b)", "a"),
                 (r"(?=(?:a|a)*b)a", "a"), (r"(?=\w*(?=\w*q))a", "a")]


def test_le5_lookarounds_are_read_in_linear_work():
    _lp, le, restore = _loaded()
    try:
        for pattern, unit in _LE5_PATTERNS:
            read = le.compile(pattern)
            for call in ("search", "finditer"):
                works, _sizes = _works(read, call, lambda n, unit=unit: unit * n)
                assert _linear(works), (pattern, call, works)
            assert _answers(read, unit * 9) == _answers(re.compile(pattern), unit * 9), pattern
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE6 -- groups in positive lookarounds
# ---------------------------------------------------------------------------
_LE6_TABLE = [
    (r"(?:(?=(a)|b).)?", ["ab", "ba"]), (r"(?=(a))(b)?a", ["a"]), (r"(a)(?=(b))", ["ab"]), (r"(a)(?=x|(b))", ["ab"]),
    (r"(?<=(a))b", ["ab", "b"]), (r"\W{0,2}?(?=(?<=(\w\b)) )", ["a1\na A __", "1AaA1 __A"]),
    (r"(?=(?=(\A^){1,3}))", ["", " ", "kAk\n"]), (r"(?=(?=(a)(b)?))a", ["ab", "a"]), (r"(?=(?P<n>a)|b)\w", ["a", "b"]),
    (r"(a)(?!(b)c)", ["ab", "abc"]), (r"(?:(?!(b)c)(.))*", ["abd"]), (r"(?<=(?=(a))a)b", ["aab"]),
]


def test_le6_the_groups_of_a_positive_lookaround_are_those_re_gives():
    _lp, le, restore = _loaded()
    try:
        for pattern, texts in _LE6_TABLE:
            read = le.compile(pattern)
            for text in texts:
                assert _answers(read, text) == _answers(re.compile(pattern), text), (pattern, text)
        for pattern, why in ((r"(?:(?=(a)|b).)*", "a repeat may pass twice"), (r"(?=(\w+))", "wider than 256"),
                             (r"(?:(?<=(a))b)+", "a repeat may pass twice"), (r"(?<=(.{300}))", "wider than 256")):
            try:
                le.compile(pattern)
            except le.Unreadable as exc:
                assert why in exc.reason, (pattern, exc.reason)
            else:
                raise AssertionError(f"{pattern} is read")
        # Not blind: re keeps a group a lookaround set in an earlier pass, which the refused forms would need.
        assert re.match(r"(?:(?=(a)|b).)*", "ab").span(1) == (0, 1)
        assert le.compile(r"(?=(\w{1,256}))").search("ab").span(1) == (0, 2), "a bounded lookahead is read"
        assert le.compile(r"(?<=(.{200}))x").search("a" * 200 + "x").span(1) == (0, 200), "a bounded lookbehind too"
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE7 -- possessive repeats and atomic groups of one set
# ---------------------------------------------------------------------------
_LE7_TABLE = [(r"a*+b", ["aab", "aa"]), (r"\w{2,4}+x", ["abx", "abcdex", "abcdx"]), (r"(?>a+)b", ["aab"]),
              (r"(?>a*?)b", ["ab", "b"]), (r"(?>[ab]{1,2}?)", ["ab"]), (r"x?+x", ["x", "xx"]),
              (r"(?:a{1,3}+)a", ["aaaa", "aa"]), (r"(?>x)y", ["xy"]), (r"[ab]++b", ["abb"])]


def test_le7_a_possessive_repeat_or_an_atomic_group_of_one_set_is_read_exactly():
    _lp, le, restore = _loaded()
    try:
        for pattern, texts in _LE7_TABLE:
            read = le.compile(pattern)
            for text in texts:
                assert _answers(read, text) == _answers(re.compile(pattern), text), (pattern, text)
        for pattern, why in ((r"(?:ab)*+", "possessive repeat of more than one set"),
                             (r"(?>ab|a)b", "atomic group of more than one set")):
            try:
                le.compile(pattern)
            except le.Unreadable as exc:
                assert why in exc.reason, (pattern, exc.reason)
            else:
                raise AssertionError(f"{pattern} is read")
        # Not blind: a possessive repeat gives back nothing, where a greedy one would.
        assert re.search(r"a*+a", "aaa") is None and le.compile(r"a*+a").search("aaa") is None
        assert le.compile(r"a*a").search("aaa").span() == (0, 3)
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE8 -- refused by name
# ---------------------------------------------------------------------------
def test_le8_a_back_reference_a_conditional_and_a_bytes_pattern_are_refused_by_name():
    _lp, le, restore = _loaded()
    try:
        for pattern, why in ((r"(a)\1", "a back-reference"), (r"(?P<x>a)(?P=x)", "a back-reference"),
                             (r"(a)?(?(1)b|c)", "a conditional"), (r"(?:((?(1)x|a))-)+", "a conditional"),
                             (rb"a+", "only a text pattern")):
            try:
                le.compile(pattern)
            except le.Unreadable as exc:
                assert isinstance(exc, re.error) and why in exc.reason and why in str(exc), (pattern, str(exc))
            else:
                raise AssertionError(f"{pattern!r} is read")
        # Not blind: re reads the conditional witness from a path it left (a-x- has no match, a-a- has one).
        assert re.fullmatch(r"(?:((?(1)x|a))-)+", "a-x-") is None
        assert re.fullmatch(r"(?:((?(1)x|a))-)+", "a-a-") is not None
        try:
            le.compile("(")
        except le.Unreadable:
            raise AssertionError("an invalid pattern is invalid, not unreadable") from None
        except re.error:
            pass
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE9 -- bounded cost of a pattern
# ---------------------------------------------------------------------------
def test_le9_what_a_pattern_may_cost_the_engine_is_bounded_and_refused_at_once():
    _lp, le, restore = _loaded()
    try:
        many_sets = "".join(re.escape(chr(0x100 + k)) for k in range(le.MOST_SETS + 1))
        many_reads = "".join(f"[^{chr(0x41 + k)}]" for k in range(40))
        for pattern, why in ((r"(?:a{1,100}){1,100}", f"more than {le.MOST_INSTRUCTIONS} steps"),
                             (r"(?:){4294967294}", f"more than {le.MOST_INSTRUCTIONS} steps"),
                             (r"(?:){2500}a", f"more than {le.MOST_INSTRUCTIONS} steps"),
                             (r"(?:(?:){65535}){65535}", f"more than {le.MOST_INSTRUCTIONS} steps"),
                             (r"(?!(){100000})", f"more than {le.MOST_INSTRUCTIONS} steps"),
                             (many_sets, f"more than {le.MOST_SETS} sets"), (many_reads, "passes over every code")):
            started = time.process_time()
            try:
                le.compile(pattern)
            except le.Unreadable as exc:
                assert why in exc.reason, (pattern[:40], exc.reason)
            else:
                raise AssertionError(f"{pattern[:40]} is read")
            assert time.process_time() - started < 2.0, ("refused at once", pattern[:40])
        # Not blind: each bound is near what a pattern may hold, one less is read.
        assert le.compile(r"(?:a{1,20}){1,20}").search("a" * 50) is not None
        assert le.compile(r"(?:){1500}a").search("ba").span() == (1, 2), "rounds of an empty body under the bound"
        fewer = "".join(chr(0x100 + k) for k in range(le.MOST_SETS))
        assert le.compile(re.escape(fewer)).search("x" + fewer).span() == (1, 1 + le.MOST_SETS)
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE10 -- compile_linear dispatches
# ---------------------------------------------------------------------------
def test_le10_compile_linear_gives_re_the_linear_the_engine_the_rest_and_refuses_what_neither_reads():
    lp, le, restore = _loaded()
    try:
        linear = lp.compile_linear(r"\bword\b", re.I)
        assert isinstance(linear, re.Pattern) and linear.pattern == r"\bword\b", "a linear pattern runs under re"
        costly = lp.compile_linear(r"(a+)+b")
        assert isinstance(costly, le.Pattern), type(costly)
        assert costly.search("aaaa!aab").span() == (5, 8), "the engine reads it"
        for pattern, why in ((r"(a)\1+x", "a back-reference"), (r"(a)?(?(1)b|c)", "a conditional")):
            try:
                lp.compile_linear(pattern)
            except lp.CostlyPattern as exc:
                assert "pattern refused:" in str(exc) and why in str(exc), (pattern, str(exc))
            else:
                raise AssertionError(f"{pattern} is read")
        try:
            lp.compile_linear(r"(a+)+b", engine=False)
        except lp.CostlyPattern as exc:
            assert str(exc) == "pattern refused: a text can make it take exponential time", str(exc)
        else:
            raise AssertionError("with the engine left out, a costly pattern is refused as before")
        try:
            lp.compile_linear("(")
        except lp.CostlyPattern:
            raise AssertionError("an invalid pattern is refused as invalid") from None
        except re.error:
            pass
    finally:
        restore()
    # Where the engine cannot be loaded, the check refuses as before, and says why.
    loaded, restore = isolate(targets={"opti_oignon.linear_pattern": source("linear_pattern.py")},
                              blocked=("opti_oignon.linear_engine",))
    try:
        lp = loaded["opti_oignon.linear_pattern"]
        try:
            lp.compile_linear(r"(a+)+b")
        except lp.CostlyPattern as exc:
            assert "cannot be loaded" in str(exc), str(exc)
        else:
            raise AssertionError("a costly pattern is read where the engine cannot be loaded")
        assert isinstance(lp.compile_linear(r"\bword\b"), re.Pattern), "control: a linear pattern still runs"
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE11 -- templates, functions and counts
# ---------------------------------------------------------------------------
_LE11_TEMPLATES = [r"<\1>", r"\g<name>-\g<0>", r"\g<2>\n\t\\", "plain", "", r"[\1|\2]", r"\0\08x", r"\101"]


def test_le11_a_template_a_function_and_a_count_are_taken_as_re_takes_them():
    _lp, le, restore = _loaded()
    try:
        pattern, text = r"(?P<name>a+)|(b)", "aab-b-ab"
        compiled, read = re.compile(pattern), le.compile(pattern)
        for template in _LE11_TEMPLATES:
            for count in (0, 1, 2, -1):
                assert read.subn(template, text, count) == compiled.subn(template, text, count), (template, count)
            assert read.match(text).expand(template) == compiled.match(text).expand(template), template
        def call(found):
            return None if found.group(2) else found.group().upper()
        assert read.sub(call, text) == compiled.sub(call, text)
        for template in (r"\3", r"\g<other>", r"\g<1", "\\"):
            try:
                compiled.sub(template, text)
            except (re.error, IndexError) as exc:
                want = (type(exc), str(exc))
            try:
                read.sub(template, text)
            except (re.error, IndexError) as exc:
                assert (type(exc), str(exc)) == want, (template, str(exc), want)
            else:
                raise AssertionError(f"{template} is expanded")
        # Not blind: an unmatched group expands to nothing, and a negative count replaces nothing.
        assert compiled.sub(r"<\2>", "a") == "<>" and compiled.subn("x", text, -1) == (text, 0)
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE12 -- the match and the pattern
# ---------------------------------------------------------------------------
def test_le12_the_match_and_the_pattern_answer_as_re_s_do():
    _lp, le, restore = _loaded()
    try:
        pattern = r"(?P<first>a)(b)?(?P<third>c)?"
        compiled, read = re.compile(pattern, re.I), le.compile(pattern, re.I)
        assert (read.pattern, read.flags, read.groups, dict(read.groupindex)) == \
            (compiled.pattern, compiled.flags, compiled.groups, dict(compiled.groupindex))
        assert type(read.flags) is type(compiled.flags) is int, "the flags are a number, as re gives them"
        want, got = compiled.search("xxAc", 1, 4), read.search("xxAc", 1, 4)
        for call in (lambda m: m.group(), lambda m: m.group(1, "third", 0), lambda m: m["first"], lambda m: m[2],
                     lambda m: m.groups(), lambda m: m.groups("-"), lambda m: m.groupdict(),
                     lambda m: m.groupdict("-"), lambda m: [m.span(g) for g in (0, 1, 2, "third")],
                     lambda m: (m.start(2), m.end(2), m.start("third")), lambda m: m.regs,
                     lambda m: (m.lastindex, m.lastgroup, m.pos, m.endpos, m.string), lambda m: bool(m),
                     lambda m: m.expand(r"\g<first>|\2|\3")):
            assert call(got) == call(want)
        for bad in (4, -1, "missing", 1.5):
            for read_one in (lambda m, bad=bad: m.group(bad), lambda m, bad=bad: m.span(bad)):
                try:
                    read_one(want)
                except IndexError:
                    pass
                try:
                    read_one(got)
                except IndexError:
                    continue
                raise AssertionError(f"group {bad!r} is read")
        for text in ("a-abc-ab", "", "bbb"):
            assert read.findall(text) == compiled.findall(text)
            for most in (0, 1, -1):
                assert read.split(text, most) == compiled.split(text, most), (text, most)
        assert le.compile("a").findall("aa") == ["a", "a"] and le.compile("(a)x").findall("axax") == ["a", "a"]
        for wrong in (b"abc", 3):
            try:
                read.search(wrong)
            except TypeError:
                continue
            raise AssertionError(f"{wrong!r} is searched")
        assert got.re is read and want.re is compiled, "a match names its pattern"
        # A compiled pattern is a value, as re's is: a copy is itself, and it pickles as the call that makes it.
        assert copy.copy(read) is read and copy.deepcopy(read) is read and copy.deepcopy(got) is got
        assert copy.copy(compiled) is compiled and copy.deepcopy(compiled) is compiled, "control: as re's"
        again = pickle.loads(pickle.dumps(read))
        assert again is not read and _answers(again, "xxAc a ABc") == _answers(compiled, "xxAc a ABc")
        assert repr(read) == f"linear_engine.compile({pattern!r}, {compiled.flags})"
        assert read == le.compile(pattern, re.I) and hash(read) == hash(le.compile(pattern, re.I))
        assert read != le.compile(pattern) and compiled == re.compile(pattern, re.I), "control: as re's"
        for call in (lambda r: r.search("a", 0, None), lambda r: r.sub("x", "a", 1.5), lambda r: r.split("a", 1.5),
                     lambda r: r.subn("x", "a", "1")):
            for engine in (compiled, read):
                try:
                    call(engine)
                except TypeError:
                    continue
                raise AssertionError(f"{engine!r} takes what re refuses")
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE13 -- windows
# ---------------------------------------------------------------------------
def test_le13_the_windows_of_the_backward_pass_change_no_answer():
    _lp, le, restore = _loaded()
    default, real = le.WINDOW, le._Alive.at
    reads = []

    def counted(alive, at):
        if (at - alive.start) // alive.size != alive.window:
            reads.append(alive.start)
        return real(alive, at)

    le._Alive.at = counted
    try:
        corpus, _refused = _corpus(le, 1413, 150, 4, 12)
        windows = {}
        for window in (default, 1, 2, 3):
            le.WINDOW = window
            reads.clear()
            for *_head, read, _cases in corpus:
                read._last = None          # the passes of the texts read under the last size are not kept
            checks, divergences, kinds = _differ(corpus, whole_every=2)
            assert not divergences and checks == 600, (window, divergences)
            windows[window] = len(reads)
        # Not blind: the machine reaches positions past the first window, which are read again from their mark.
        assert windows[1] > 2 * windows[default] > 0, windows
        assert kinds["several"] > 50 and kinds["matched"] > 200, kinds
    finally:
        le.WINDOW, le._Alive.at = default, real
        restore()


# ---------------------------------------------------------------------------
# LE14 -- the cache bound
# ---------------------------------------------------------------------------
def _caches(read):
    programs = [read._main] + [program for look in read._shared.looks for program in (look.scan, look.exact)
                               if program is not None]
    return [cache for program in programs for cache in (program.cache, getattr(program, "final_cache", None))
            if cache is not None]


def test_le14_a_cache_of_one_to_three_moves_changes_no_answer_and_none_holds_more_than_its_bound():
    _lp, le, restore = _loaded()
    try:
        default = le.MOST_MOVES
        widest = 0
        for bound in (1, 2, 3, default):
            le.MOST_MOVES = bound
            corpus, _refused = _corpus(le, 1414, 150, 4, 12)
            checks, divergences, _kinds = _differ(corpus, whole_every=2)
            assert not divergences and checks == 600, (bound, divergences)
            held = max(len(cache.moves) for *_head, read, _cases in corpus for cache in _caches(read))
            assert held <= bound, (bound, held)
            if bound == default:
                widest = held
        # Not blind: unbounded, the caches of this corpus hold more than three moves.
        assert widest > 3, widest
    finally:
        le.MOST_MOVES = default
        restore()


# ---------------------------------------------------------------------------
# LE15 -- work per character and the size of the program
# ---------------------------------------------------------------------------
def _fresh(text):
    """The same text in a new object: a pattern keeps the passes of the last text it read, by identity."""
    return (text + "!")[:-1]


def test_le15_the_work_per_character_is_bounded_by_the_program_and_does_not_grow_where_its_states_repeat():
    _lp, le, restore = _loaded()
    try:
        words = ["alpha", "bravo", "delta", "echo", "golf", "hotel", "india", "juliet", "kilo", "lima", "mike",
                 "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray",
                 "yankee", "zulu", "amber", "basalt", "cobalt", "dune", "ember", "flint", "garnet"]
        text = "lorem ipsum dolor sit amet, consectetur adipiscing elit " * 400
        works = {}
        for count in (2, 30):
            read = le.compile("(?i)(?:" + "|".join(words[:count]) + r")\d")
            assert read._needles is None, "the prefilter does not answer: the whole text is read"
            read.search(_fresh(text))
            read.work = 0
            assert read.search(_fresh(text)) is None
            works[count] = (read.work, len(read._main.code))
        assert works[30][1] > 6 * works[2][1], ("the program grows", works)
        assert works[30][0] <= 1.25 * works[2][0], works
        assert works[2][0] >= len(text), ("the work counts every character", works)
        # Where every position may be a new state (x then any K characters then c), the work per character is
        # bounded by the program's size: it grows with K, never past a constant times it.
        rng = random.Random(15)
        noise = "".join(rng.choice("cy") for _ in range(6000)) + "x"
        per = {}
        for width in (10, 100, 400):
            read = le.compile(r"x.{%d}c" % width)
            read.work = 0
            assert read.search(_fresh(noise)) is None
            per[width] = (read.work / len(noise), len(read._main.code))
        assert all(work <= 8 * size for work, size in per.values()), per
        assert per[400][0] > 4 * per[10][0], ("not blind: these states do not repeat", per)
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE16 -- the prefilter
# ---------------------------------------------------------------------------
def test_le16_the_prefilter_is_sound():
    _lp, le, restore = _loaded()
    try:
        corpus, _refused = _corpus(le, 1416, 600, 6, 9)
        answered = matched = 0
        for pattern, flags, compiled, read, cases in corpus:
            if read._needles is None:
                continue
            for text, pos, endpos in cases:
                end = len(text) if endpos is None else min(endpos, len(text))
                start = min(pos, len(text))
                if not any(text.find(needle, start, end) >= 0 for needle in read._needles):
                    answered += 1
                    assert compiled.search(text, pos, end) is None, (pattern, text, read._needles)
                for found in compiled.finditer(text, pos, end):
                    matched += 1
                    assert any(needle in found.group() for needle in read._needles), (pattern, text)
        assert answered > 100 and matched > 100, (answered, matched)
        # A class between two literals is no string of its own: the needles hold no join across it.
        # A letter under IGNORECASE is no string either: its other case matches too.
        for pattern in (r"a.b", r"a\sb", r"x[ab]y", r"k\dx", r"(?:ab|a.)c", r"a\Wb", r"(?s)a.b", r"a[^b]b", r"(?i)abc",
                        r"(?i)k\dx", r"(?i:ab)c", r"x(?i:y)"):
            read, compiled = le.compile(pattern), re.compile(pattern)
            for text in ("a b", "axb", "xay", "k1x", "abc", "a-c", "a.c", "a\nb", "aab", "a1b", "ABC", "K1X", "abC", "XY"):
                found = compiled.search(text)
                assert (read.search(text) is None) == (found is None), (pattern, text)
                if found is not None and read._needles is not None:
                    assert any(needle in found.group() for needle in read._needles), (pattern, text, read._needles)
        # A string past the bound is cut, never shortened: the needles of a long join are its whole parts.
        needles = le.compile(r"(?:short|x{32})y")._needles
        assert needles is None or all(len(needle) <= le.MOST_LENGTH for needle in needles), needles
        assert le.compile(r"(?:short|x{32})y").search("x" * 32 + "y") is not None
        assert le.compile(r"(?:short|x{31})y").search("x" * 32 + "y").span() == (1, 33)
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE17 -- the memory of the backward pass
# ---------------------------------------------------------------------------
def test_le17_the_backward_pass_keeps_one_set_a_window_whatever_the_length():
    _lp, le, restore = _loaded()
    real = le._Alive
    made = []

    class Watched(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    le._Alive = Watched
    try:
        read = le.compile(r"b[^c]*c|b")
        for size in (5_000, 50_000):
            made.clear()
            found = read.findall("b" * size)
            assert len(found) == size, size
            main = [alive for alive in made if alive.program is read._main]
            assert len(main) == 1, len(main)
            assert len(main[0].marks) <= size // le.WINDOW + 2, (size, len(main[0].marks))
            width = read._main.width
            assert width == len(read._main.code) // 8 + 1
            assert all(type(state) is bytes and len(state) == width for state in main[0].marks.values()), \
                "a kept set costs one bit an instruction"
            assert main[0].sets is not None and len(main[0].sets) <= le.WINDOW + 1, "one window at a time"
    finally:
        le._Alive = real
        restore()


# ---------------------------------------------------------------------------
# LE18 -- versions
# ---------------------------------------------------------------------------
def test_le18_a_boundary_in_an_empty_text_follows_the_interpreter_and_an_unmeasured_version_is_refused():
    _lp, le, restore = _loaded()
    try:
        for pattern in (r"\b", r"\B", r"\B\B", r"(?a)\B", r"x?\B"):
            read, compiled = le.compile(pattern), re.compile(pattern)
            for text, pos, endpos in (("", 0, None), ("ab", 0, 0), ("ab", 2, 1), ("a", 1, None)):
                assert _answers(read, text, pos, endpos, False) == _answers(compiled, text, pos, endpos, False), \
                    (pattern, text, pos, endpos)
        assert le.MEASURED == _MEASURED, "the versions these contracts skip on are the engine's own"
        measured = le.MEASURED
        le.MEASURED = tuple(version for version in measured if version != sys.version_info[:2])
        try:
            le.compile("a+")
        except le.Unreadable as exc:
            assert "not measured" in exc.reason and f"{sys.version_info[0]}.{sys.version_info[1]}" in exc.reason
        else:
            raise AssertionError("a version the engine was not measured on is read")
        finally:
            le.MEASURED = measured
        assert sys.version_info[:2] in measured, "control: the running version is one measured"
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE19 -- one read at a time
# ---------------------------------------------------------------------------
def test_le19_one_read_of_a_pattern_at_a_time():
    _lp, le, restore = _loaded()
    real_completes, default = le._completes, le.MOST_MOVES
    inside, resume, done = threading.Event(), threading.Event(), threading.Event()
    first, answers = [], {}

    def held(*args, **kwargs):
        if not first and threading.current_thread().name == "first":
            first.append(True)
            inside.set()
            resume.wait(2.0)
        return real_completes(*args, **kwargs)

    def run(name, text):
        answers[name] = [found.span() for found in read.finditer(text)]
        if name == "second":
            done.set()

    le._completes, le.MOST_MOVES = held, 2
    try:
        read = le.compile(r"(?:a|b)*c|a")
        one = threading.Thread(target=run, args=("first", "abab" * 20), name="first")
        one.start()
        assert inside.wait(2.0), "the first read builds a move"
        two = threading.Thread(target=run, args=("second", "ba" * 30 + "c"), name="second")
        two.start()
        waited = done.wait(0.3)
        resume.set()
        one.join(5.0)
        two.join(5.0)
        assert not waited, "the second read ran while the first was building a move"
        assert answers["first"] == [m.span() for m in re.finditer(r"(?:a|b)*c|a", "abab" * 20)]
        assert answers["second"] == [m.span() for m in re.finditer(r"(?:a|b)*c|a", "ba" * 30 + "c")]
        for cache in _caches(read):
            for key, number in cache.moves.items():
                following = cache.states[key[0]]
                members = None if key[1] is None else read._shared.infos[key[1]][0]
                assert real_completes(read._main, following, members, key[2], key[3]) == cache.states[number]
        # Not blind: the second read waited for the first, then answered.
        assert done.is_set() and len(answers) == 2
    finally:
        resume.set()
        le._completes, le.MOST_MOVES = real_completes, default
        restore()


# ---------------------------------------------------------------------------
# LE20 -- characters beyond ASCII
# ---------------------------------------------------------------------------
def test_le20_characters_beyond_ascii_answer_as_re_s_do():
    _lp, le, restore = _loaded()
    try:
        chars = [chr(point) for point in (0xE9, 0x212A, 0xDF, 0x130, 0x3A3, 0x3C3, 0x3C2, 0x663, 0xA0, 0x2028, 0x85,
                                          0x1F600, 0x5F, 0x31)]
        texts = ["".join(chars), "k" + chr(0x212A) + "K", chr(0x3C3) + " " + chr(0x3A3) + chr(0x3C2) + "\n",
                 chr(0xE9) + "t" + chr(0xE9) + "\n", chr(0x663) + "1", chr(0x2028) + "a" + chr(0x85)]
        patterns = [r"\w+", r"(?a)\w+", r"\b\w", r"(?a)\b\w", r"(?i)k", "(?i)" + chr(0x3C3) + "+", r"\d+", r"(?a)\d",
                    r"\s+", r"(?m)\w$", r"\w$", r"(?i)[^k]+", r"(?i)" + chr(0xDF), r"\B.", r".+"]
        for pattern in patterns:
            read, compiled = le.compile(pattern), re.compile(pattern)
            for text in texts:
                assert _answers(read, text) == _answers(compiled, text), (ascii(pattern), ascii(text))
        # Not blind: case folding and the Unicode classes reach beyond ASCII here.
        assert re.findall(r"(?i)k", texts[1]) == ["k", chr(0x212A), "K"]
        assert re.match(r"\w", chr(0x663)) is not None and re.match(r"(?a)\w", chr(0x663)) is None
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE21 -- a match past endpos
# ---------------------------------------------------------------------------
_LE21_PATTERNS = [r"$", r"(?m)$", r"\Z", r"\A", r"^", r"(?m)^", r"\b", r"\B", r"(?a)\b", r"x*", r"(?:x)*", r"(?:)",
                  r"(?=)", r"(?!x)", r"(?<=a)", r"(?<!a)", r"(?=$)", r"(?=(\Z))", r"(?:$|a)", r"()", r"(?:x|)$",
                  r"(?<=\n)$", r"(?!)", r"$\Z", r"(?m)$(?=\n)", r"\b(?=)", r"a??", r"(?:a*)?", r"(a)*", r"(?:(a))*",
                  r"(a)?", r"(?:(?:a))*", r"(?=(a)?)", r"(?:a|b)*", r"[ab]*?", r"(?P<n>x)*"]


def test_le21_a_match_past_endpos_answers_as_re_s_does():
    _lp, le, restore = _loaded()
    try:
        seen = 0
        for pattern in _LE21_PATTERNS:
            read, compiled = le.compile(pattern), re.compile(pattern)
            for text in ("", "a", "a\n", "\n\n", "ab\n", "x", "a b"):
                for pos in range(len(text) + 2):
                    for endpos in range(0, pos):
                        want, got = _regs(compiled.match(text, pos, endpos)), _regs(read.match(text, pos, endpos))
                        assert got == want, (pattern, text, pos, endpos, want, got)
                        seen += want is not None
        # Not blind: re finds matches past endpos for some and none for others (a repeat of one character fails).
        assert seen > 100, seen
        assert re.compile(r"(?:)").match("ab", 2, 1) is not None and re.compile(r"x*").match("ab", 2, 1) is None
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE22 -- the threads of one position
# ---------------------------------------------------------------------------
_LE22_TABLE = [(r"((a|)*)*", "aaaa"), (r"(?:(?:a|)*b|)*c", "abab c"), (r"((?:(a|)*|b)*)*x", "ababx"),
               (r"(?:(?:(?:a|)*)*)*", "aaa"), (r"(\w+\s?)*$", "a a a!"), (r"(?:x*)*?y", "xxxy"),
               (r"(a|)+?(b|)*?c", "abc"), (r"(?:(?=a)|b|)*a", "baba")]


def test_le22_a_position_holds_at_most_m_times_d_plus_one_threads():
    _lp, le, restore = _loaded()
    real = le.Pattern._close
    peak = [0]

    def watched(self, program, pc, j, caps, at, end, bits, seen, out):
        real(self, program, pc, j, caps, at, end, bits, seen, out)
        peak[0] = max(peak[0], len(seen))

    le.Pattern._close = watched
    try:
        copies = 0
        for pattern, text in _LE22_TABLE:
            read = le.compile(pattern)
            peak[0] = 0
            assert [found.regs for found in read.finditer(text)] == [found.regs for found in re.finditer(pattern, text)]
            size, depth = len(read._main.code), max(read._main.depth)
            assert 0 < peak[0] <= size * (depth + 1), (pattern, peak[0], size, depth)
            copies += peak[0] > size
        # Not blind: the copies for the empty rounds are there, more states than instructions at one position.
        assert copies >= 2, copies
    finally:
        le.Pattern._close = real
        restore()


# ---------------------------------------------------------------------------
# LE23 -- the cases a pass of mutants found unheld
# ---------------------------------------------------------------------------
# (pattern, flags, text, pos, endpos): each one where a mutant of the engine that every other contract let live
# answered otherwise than re (found by running the differential against each surviving mutant).
_LE23_WITNESSES = [
    ('(?<!\\w{2})(?<!)-?', 0, '1\n\naba', 0, None),
    ('[a-z]{1,3}+[ab]{2,}(?=\\b)\\A', 16, ' b', 2, 1),
    ('[^a]+?(?:\\Z.++\\A|(?:(\\w?+^){0,2}?\\W??x??(?<=(?:ab|xk)(?:\\w|-))|.{2,}[a-z]{2,})+? {0,3}(?:\\b\\w{2}(?>a*)|[^a]+.)[^a]++)?', 0, 'xk_ b\nk', 2, 6),
    ('(?>[ab]{1,2}?)(?<=((?:ab|xk)))', 16, '_b\nbab', 0, 9),
    ('(\\B\\W??b+?((?:\\B b{0,3}|(?>x)){0,3}\\W??){0,3}){1,2}?', 8, '-_b ', 2, None),
    ('\\w*(?:(?:\\s{1,3}(?>[ab]{1,2}?)(?:-+?\\Z|\\B\\Z[^a]*k*){1,2}?|$-{2}(?=-{1,3}\\B))?(?P<g1>\\A)| {1,3}((?:\\b[^a]{1,2}?(?>x)|\\A\\Bx{0,2}?[a-z])*\\d?){0,3}){0,2}?', 24, ' A1kx-Akx', 0, None),
    ('\\b( ?b??){1,3}\\Z *+', 16, 'k\nx_k-\nb', 0, 11),
    ('(?<!)\\d?+', 10, 'k\n_a 1Aa\n', 0, None),
    ('(?:(?:[^a]?+|(?:b\\A\\W?+k{2}|\\w?+)+?b*\\B\\s{2,}){0,3}((?!$^[^a]{1,3}-))|[ab]{0,2}?(?:.??|\\W??)*?\\W++){1,3}', 10, 'a\nx_x11 \n', 2, None),
    ('-??.(?<=(?:ab|xk))', 256, '1ababAa', 2, None),
    ('(?<!(((?:ab|xk))))', 2, '\nab1-_', 2, None),
    ('(?<=.{2}(?:ab|xk))b*?', 256, ' baab ', 0, 5),
    ('((?is:[^a])??(?:([a-z]??\\s{0,2}?)+?k+?|\\B(?i:[ab]*?\\s{0,2}?)*){0,3}){0,3}', 256, 'bAAa', 0, None),
    ('\\B\\A\\A', 24, 'x\n', 2, 1),
    ('(?<=(?:ab|xk))\\W*+', 24, 'abbk', 0, None),
    ('(?P<g1>(\\W?? {0,2}?\\d?+\\b){1,3}.\\Z){0,2}?', 8, '-\n_A k ab', 1, 8),
    ('(?=(\\b)\\b)(?!\\W\\Bb??(\\A(?=\\B)x{1,3}(?:k*|k+?\\wb{1,3}+\\w){2,}))', 2, '\n1', 2, 1),
]


def test_le23_the_cases_a_pass_of_mutants_found_unheld_answer_as_re_does():
    _lp, le, restore = _loaded()
    try:
        assert len(_LE23_WITNESSES) == 17, "control: every witness is held"
        for pattern, flags, text, pos, endpos in _LE23_WITNESSES:
            read, compiled = le.compile(pattern, flags), re.compile(pattern, flags)
            assert _answers(read, text, pos, endpos) == _answers(compiled, text, pos, endpos), (pattern, text, pos)
        # Not blind: the witnesses reach lookbehinds, counted repeats and the place past the end.
        assert sum("(?<" in pattern for pattern, *_ in _LE23_WITNESSES) >= 5
        assert sum(endpos is not None and pos > endpos for *_, pos, endpos in _LE23_WITNESSES) >= 2
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE24 -- where a search tries a start
# ---------------------------------------------------------------------------
_LE24_PATTERNS = [(r"(?a:\S)", 0), (r"(?a:\W)", 0), (r"(?a:\D)", 0), (r"(?u:\w)", re.ASCII), (r"((?a:\s))x?", 0),
                  (r"(?a:[\S\d])", 0), (r"(?:(?a:\W))+", 0), (r"(?a:\w)", 0), (r"\S", 0), (r"(?a:[^\s])", 0),
                  (r"(?u:\d)+", re.ASCII), (r"(?a:\S)|y", 0), (r"(?i:\S)", 0)]


def test_le24_a_search_tries_a_start_where_re_s_search_does():
    _lp, le, restore = _loaded()
    try:
        chars = "".join(chr(point) for point in (0x85, 0xE9, 0x660, 0xA0, 0x2028, 0x3000, 0x1D7CE))
        texts = [chars, "\x85b", "\xe9!", "\u0660x", "!\xe9", "a" + chars + " b", " \x85 ", "x\u3000y"]
        differs = 0
        for pattern, flags in _LE24_PATTERNS:
            read, compiled = le.compile(pattern, flags), re.compile(pattern, flags)
            for text in texts:
                assert _answers(read, text) == _answers(compiled, text), (ascii(pattern), ascii(text))
                found, first = compiled.search(text), compiled.match(text)
                differs += first is not None and (found is None or found.span() != first.span())
        # Not blind: re's search skips a start its own match takes, and the engine's search does the same.
        assert differs >= 3, differs
        assert re.search(r"(?a:\S)", "\x85b").span() == (1, 2) and re.match(r"(?a:\S)", "\x85b").span() == (0, 1)
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE25 -- one match after another from a moving position
# ---------------------------------------------------------------------------
def _tokens(read, text, call):
    spans, pos = [], 0
    while pos < len(text):
        found = getattr(read, call)(text, pos)
        if found is None:
            break
        spans.append(found.span())
        pos = found.end() if found.end() > pos else pos + 1
    return spans


def test_le25_a_reader_that_takes_one_match_after_another_reads_the_text_in_linear_work():
    lp, le, restore = _loaded()
    try:
        pattern = r"(?:a|ab)*c|\w+|\s+|."
        read = lp.compile_linear(pattern)
        assert isinstance(read, le.Pattern), "witness: the check refuses it, the engine reads it"
        assert _tokens(read, "ab cd ab c ", "match") == _tokens(re.compile(pattern), "ab cd ab c ", "match")
        for call in ("match", "search"):
            works = []
            for size in (2000, 4000, 8000):
                text = "ab cd " * (size // 6)
                read.work = 0
                spans = _tokens(read, text, call)
                assert len(spans) > size // 6, (call, len(spans))
                works.append(read.work)
            assert _linear(works), (call, works)
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE26 -- a subclass of str
# ---------------------------------------------------------------------------
def test_le26_a_subclass_of_str_is_read_as_its_exact_text_and_runs_no_code_during_a_read():
    _lp, le, restore = _loaded()
    calls = []

    class Upper(str):
        def __getitem__(self, key):
            calls.append(key)
            return str.__getitem__(self, key).upper()

        def __len__(self):
            calls.append("len")
            return 0

    try:
        pattern = r"(?:a|ab)*c"
        read, compiled = le.compile(pattern), re.compile(pattern)
        given = Upper("xabc abc")
        found = read.search(given)
        assert found.span() == compiled.search(given).span() == (1, 4) and found.string is given
        assert type(found.group()) is str and found.group() == "abc"
        assert read.findall(given) == compiled.findall(given) and read.sub("-", given) == compiled.sub("-", given)
        assert read.split(given) == compiled.split(given)
        assert calls == [], ("none of the subclass's methods ran", calls[:5])
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE27 -- finditer is lazy
# ---------------------------------------------------------------------------
def test_le27_finditer_is_lazy_and_other_reads_between_two_steps_change_nothing():
    _lp, le, restore = _loaded()
    try:
        read = le.compile(r"b[^c]*c|b")
        text = "b" * 40000
        whole = read.findall(_fresh(text))
        read.work = 0
        steps = read.finditer(_fresh(text))
        first = next(steps)
        one = read.work
        read.work = 0
        list(read.finditer(_fresh(text)))
        assert first.span() == (0, 1) and len(whole) == 40000
        assert one * 2 < read.work, ("a step does not read every match", one, read.work)
        # Other reads of the same pattern between two steps change nothing in what the steps find.
        mixed, steps, other = [], read.finditer("bxbcbb"), "cbbcxb"
        for found in steps:
            mixed.append(found.span())
            read.search(other)
            read.findall(_fresh(other))
        assert mixed == [found.span() for found in re.finditer(r"b[^c]*c|b", "bxbcbb")]
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE28 -- the cases the last pass of mutants found unheld
# ---------------------------------------------------------------------------
_LE28_WITNESSES = [
    (r"((a*)(?=))*", 0, "a", 0, None),
    (r"([^a]?\B){,}", 0, "b\u212a", 0, None),
    (r"()(?=((\)))?)", 0, "", 0, None),
    (r"(?<!(?:ab|xk)) ??", re.IGNORECASE, "xkbAk", 2, 4),
    (r"a??-?(?:[a-z]*?|\D)+", re.IGNORECASE | re.MULTILINE, "\u212a\xdfbbA\xdfax", 0, None),
    (r"(?:[ab]*?|a(?>a*)){2,}", re.MULTILINE | re.DOTALL, "\nbb\x85x-\x85\x851", 0, 12),
    (r"(?i:(?x:(\Z[a-z]{2})??\W{0,2}?){1,3}[ab]{0,3}){2,}", re.IGNORECASE, "\u2028\u212a\xdf- \u0660", 2, 5),
    (r"(?=(?u:(a)))a", 0, "a", 0, None),
    (r"(?<=(?i:(a)))b", 0, "ab", 0, None),
    (r"(?s:])?", 0, "k", 1, 0),
    (r"((?=(){,}?))", 0, "\n\u212axA", 5, 3),
    (r"(?!(|s))", 0, " \u212a", 2, 1),
]


def test_le28_the_cases_the_last_pass_of_mutants_found_unheld_answer_as_re_does():
    _lp, le, restore = _loaded()
    try:
        assert len(_LE28_WITNESSES) == 12, "control: every witness is held"
        for pattern, flags, text, pos, endpos in _LE28_WITNESSES:
            read, compiled = le.compile(pattern, flags), re.compile(pattern, flags)
            assert _answers(read, text, pos, endpos) == _answers(compiled, text, pos, endpos), (ascii(pattern), pos)
        # A lookbehind whose body takes rounds that read nothing: its forward pass takes each instruction once a
        # position, and ends.
        for pattern, text in ((r"(?<=(?:\b)*a)b", "ab" * 50), (r"(?<!(?:^|\B)+x)y", "xy y" * 20)):
            read, compiled = le.compile(pattern), re.compile(pattern)
            assert _answers(read, text) == _answers(compiled, text), pattern
        # A refusal is a re.error with what one carries: its message, and no pattern or position of a syntax error.
        try:
            le.compile(r"(a)\1")
        except re.error as exc:
            assert exc.msg == str(exc) == f"pattern not read: {exc.reason}", exc.msg
            assert exc.pattern is None and exc.pos is None
        else:
            raise AssertionError("a back-reference is read")
        # Not blind: the witnesses reach a group that may read nothing in a repeat, and lookarounds both ways.
        assert sum("(?=" in pattern for pattern, *_ in _LE28_WITNESSES) >= 2
        assert sum("(?<" in pattern for pattern, *_ in _LE28_WITNESSES) >= 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# LE29 -- the shapes a generator long missed
# ---------------------------------------------------------------------------
def test_le29_the_shapes_a_generator_long_missed_answer_as_re_does():
    _lp, le, restore = _loaded()
    try:
        corpus, refused = _corpus(le, 2901, 1500, 6, 8, wide=True)
        checks, divergences, kinds = _differ(corpus)
        assert not divergences, divergences
        assert checks == 1500 * 6, checks
        differ = [(pattern, flags) for pattern, flags, compiled, read, _cases in corpus
                  if (read.flags, read.groups, dict(read.groupindex))
                  != (compiled.flags, compiled.groups, dict(compiled.groupindex))]
        assert not differ, differ[:3]
        # Not blind: the corpus reaches the shapes it is about, and matches of every kind.
        held = [pattern for pattern, *_ in corpus]
        cases = [case for *_, held_cases in corpus for case in held_cases]
        assert sum(endpos is not None and pos > endpos for _text, pos, endpos in cases) > 600, "a match past endpos"
        assert sum(re.search(r"\(\?<[=!](?:[^()]|\([^()]*\))*\(\?<?[=!]", p) is not None for p in held) > 8, \
            "a lookaround nested in a lookbehind"
        assert sum(re.search(r"\((?:\?:)?(?:\\[bBAZ]|\^|\$|)\)[*+?{]", p) is not None for p in held) > 50, \
            "a counted group of one zero-width atom"
        assert all(count > 200 for count in kinds.values()), kinds
        assert refused > 0, "the generator reaches patterns the engine refuses"
    finally:
        restore()
