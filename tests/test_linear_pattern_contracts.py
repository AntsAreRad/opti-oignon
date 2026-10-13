#!/usr/bin/env python3
"""Contracts for the check of what a pattern from outside the code costs, and for the places that take one.

A regular expression written in the package is held to linear time by the
pattern cost guard, which measures it. One that comes from a model, a
configuration file or a person is not in the package: ``linear_pattern``
judges it from the structure of its automaton before it runs -- soundly,
refusing what it cannot model, never guessing -- and every place that compiles
such a pattern refuses one that may read a long text in more than linear time,
as it refuses an invalid one. The patterns the package built from its
configuration that read in more than linear time are rewritten here too, each
held to the one it replaced.
  * SL1 -- the verdicts of a table of patterns: linear, polynomial,
    exponential or unchecked; a caret in multiline mode starts every line.
  * SL2 -- no pattern of a seeded corpus that the check calls linear reads a
    long text in more than linear time, with a search or every match in turn;
    the timing oracle sees the quadratic ones.
  * SL3 -- the check's own cost is bounded: states, steps, passes over every
    code point, ways through a lookaround, depth.
  * SL4 -- the sets of characters are the engine's: case folding is read from
    it, and one shared pass gives each set what a pass of its own gives.
  * SL5 -- ``compile_linear`` returns the compiled pattern, or refuses with a
    ``re.error`` that says why; an invalid pattern is refused as before.
  * SL6 -- ``joins_exactly`` tells an exact join from one that loses matches.
  * SL7 -- the agent's grep refuses a costly pattern and tells the model what
    to search instead; a linear one still finds its lines.
  * SL8 -- the sanitizer's custom patterns: a costly one is left out with a
    warning, a linear one kept, an older red-team rule read in its line form,
    the red team's own patterns trusted as written.
  * SL9 -- the PII sanitizer's custom patterns: a costly one is left out.
  * SL10 -- the tone shifter's custom rules: a costly one is skipped.
  * SL11 -- the red team writes its fallback rule in a linear line form that
    matches as the old one did, and refuses a costly rule before writing.
  * SL12 -- the sandbox's blocked patterns are made at the first command: the
    defaults all hold, each joined one as a chain that finds what the pattern
    found; a pattern neither linear nor an exact chain refuses every command.
  * SL13 -- a chain reads a long command in linear time.
  * SL14 -- the RAG store's page cleaning reads as the patterns it replaced,
    on enumerated, structured and pinned pages, in linear time.
  * SL15 -- the address of a bound port is read as before and the header of a
    skill's section within its line, both in linear time.
  * SL16 -- a line start under multiline mode is read after every line, not
    only after blank ones.
  * SL17 -- a head that may end where another way of it reads on does not
    join exactly: a custom blocked rule built so refuses, never stops
    blocking; chains joined exactly find what their pattern finds.
  * SL18 -- a group's type flag takes the place of the pattern's, as in the
    engine.
  * SL19 -- a check holds the sets it read, whatever another check empties:
    its verdict does not hang on what earlier checks kept.
  * SL20 -- an older red-team rule whose text holds a line break keeps its
    written form.
  * SL21 -- a colon alone past the header's line still closes the header,
    as it did before.
  * SL22 -- a blocked pattern is cut at its top level only: never in a class,
    a group, a comment or after an escape; the chains of a structured corpus
    find what their pattern finds.
  * SL23 -- a range of a class is read whole.
  * SL24 -- a head that ends with a run joins only when nothing is lost:
    the part before the run ends once, the tail starts neither empty nor
    with the run's class.
  * SL25 -- a chain tries every start of its head and every place of its
    tail up to its line's break, and finds nothing where nothing matches.
  * SL26 -- every refusal of the check is a re.error, every doubt of a join
    is False; the sandbox refuses a pattern too large to read, naming it.
  * SL27 -- a repeat counted past the bound, or a lookaround longer than it,
    is refused at once.
  * SL28 -- a verdict does not hang on what earlier checks kept.
  * SL29 -- a pattern or flags of the wrong type are refused as invalid,
    never escape.
  * SL30 -- a lookaround nested in another multiplies its ways.
  * SL31 -- sets of characters written too long are refused before they
    are read.
  * SL32 -- a lookaround is judged once for a check, however often it is
    built.
  * SL33 -- a lookaround is bounded by the steps a way through it takes,
    rounds of zero-width items included.
  * SL34 -- the tone shifter applies no custom rule where the check cannot
    be loaded.
  * SL35 -- a blocked pattern whose join crosses lines is never read as a
    chain of lines.
  * SL36 -- a stripped tag is read as before after an equals sign, a slash
    or a tab.
  * SL37 -- a pattern that tries more ways from one start than the bound is
    refused.
  * SL38 -- one try of a lookaround is bounded by its ways times its steps.
  * SL39 -- every empty path to one step is a way of its own.
  * SL40 -- the lookarounds one start may try take no more than the bound
    in all.
  * SL41 -- a linear pattern a person may write for amounts is read within
    the check's bounds.
  * SL42 -- an empty round of a loop goes on past the loop, a way of its own.
  * SL43 -- the empty ways to the end are tried where the end is refused, and
    again by every match in turn.
  * SL44 -- a state where the match may end still tries every empty way it
    has, to a step or to the end, before it ends.
  * SL45 -- the lookarounds tried on those ways are bounded too.
"""
import itertools
import random
import re
import sqlite3
import sys
import time
import types
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_sl1_the_verdicts_of_a_table_of_patterns": 3.0,
    "test_sl2_no_pattern_the_check_calls_linear_reads_a_long_text_in_more_than_linear_time": 20.0,
    "test_sl3_the_check_s_own_cost_is_bounded": 6.0,
    "test_sl4_the_sets_of_characters_are_the_engine_s": 3.0,
    "test_sl5_compile_linear_returns_the_pattern_or_refuses_with_a_re_error_that_says_why": 3.0,
    "test_sl6_joins_exactly_tells_an_exact_join_from_one_that_loses_matches": 3.0,
    "test_sl7_the_agent_s_grep_refuses_a_costly_pattern_and_still_searches_a_linear_one": 6.0,
    "test_sl8_the_sanitizer_leaves_a_costly_custom_pattern_out_and_reads_an_older_red_team_rule_in_its_line_form": 6.0,
    "test_sl9_the_pii_sanitizer_leaves_a_costly_custom_pattern_out": 3.0,
    "test_sl10_the_tone_shifter_skips_a_costly_custom_rule": 3.0,
    "test_sl11_the_red_team_writes_a_linear_fallback_rule_and_refuses_a_costly_one_before_writing": 6.0,
    "test_sl12_the_sandbox_makes_its_blocked_patterns_at_the_first_command_and_refuses_what_it_cannot_hold": 15.0,
    "test_sl13_a_chain_reads_a_long_command_in_linear_time": 6.0,
    "test_sl14_the_rag_store_cleans_a_page_as_the_patterns_it_replaced_did_in_linear_time": 30.0,
    "test_sl15_a_bound_address_is_read_as_before_and_a_section_header_within_its_line": 10.0,
    "test_sl16_a_line_start_under_multiline_is_read_after_every_line_not_only_after_blank_ones": 6.0,
    "test_sl17_a_head_that_may_end_where_another_way_reads_on_does_not_join_exactly": 15.0,
    "test_sl18_a_group_s_type_flag_takes_the_place_of_the_pattern_s_as_in_the_engine": 6.0,
    "test_sl19_a_check_holds_the_sets_it_read_whatever_another_check_empties": 6.0,
    "test_sl20_an_older_red_team_rule_whose_text_holds_a_line_break_keeps_its_written_form": 3.0,
    "test_sl21_a_colon_alone_past_the_header_s_line_still_closes_the_header_as_before": 6.0,
    "test_sl22_a_blocked_pattern_is_cut_at_its_top_level_only": 10.0,
    "test_sl23_a_range_of_a_class_is_read_whole": 3.0,
    "test_sl24_a_head_that_ends_with_a_run_joins_only_when_nothing_is_lost": 3.0,
    "test_sl25_a_chain_tries_every_start_of_its_head_and_every_place_of_its_tail_to_its_line_break": 6.0,
    "test_sl26_every_refusal_of_the_check_is_a_re_error_and_every_doubt_of_a_join_is_false": 6.0,
    "test_sl27_a_repeat_counted_past_the_bound_or_a_lookaround_longer_than_it_is_refused_at_once": 6.0,
    "test_sl28_a_verdict_does_not_hang_on_what_earlier_checks_kept": 10.0,
    "test_sl29_a_pattern_or_flags_of_the_wrong_type_are_refused_as_invalid_and_never_escape": 6.0,
    "test_sl30_a_lookaround_nested_in_another_multiplies_its_ways": 6.0,
    "test_sl31_sets_of_characters_written_too_long_are_refused_before_they_are_read": 6.0,
    "test_sl32_a_lookaround_is_judged_once_for_a_check_however_often_it_is_built": 6.0,
    "test_sl33_a_lookaround_is_bounded_by_the_steps_a_way_through_it_takes_zero_width_rounds_included": 6.0,
    "test_sl34_the_tone_shifter_applies_no_custom_rule_where_the_check_cannot_be_loaded": 3.0,
    "test_sl35_a_blocked_pattern_whose_join_crosses_lines_is_never_read_as_a_chain_of_lines": 6.0,
    "test_sl36_a_stripped_tag_is_read_as_before_after_an_equals_sign_a_slash_or_a_tab": 20.0,
    "test_sl37_a_pattern_that_tries_more_ways_from_one_start_than_the_bound_is_refused": 6.0,
    "test_sl38_one_try_of_a_lookaround_is_bounded_by_its_ways_times_its_steps": 3.0,
    "test_sl39_every_empty_path_to_one_step_is_a_way_of_its_own": 6.0,
    "test_sl40_the_lookarounds_one_start_may_try_take_no_more_than_the_bound_in_all": 3.0,
    "test_sl41_a_linear_pattern_for_amounts_is_read_within_the_check_s_bounds": 6.0,
    "test_sl42_an_empty_round_of_a_loop_goes_on_past_the_loop_a_way_of_its_own": 6.0,
    "test_sl43_the_empty_ways_to_the_end_are_tried_where_the_end_is_refused_and_by_every_match_in_turn": 6.0,
    "test_sl44_a_state_where_the_match_may_end_still_tries_every_empty_way_it_has_before_it_ends": 6.0,
    "test_sl45_the_lookarounds_tried_on_the_ways_of_a_state_where_the_match_may_end_are_bounded": 6.0,
}


def _checker():
    loaded, restore = isolate(targets={"opti_oignon.linear_pattern": source("linear_pattern.py")})
    return loaded["opti_oignon.linear_pattern"], restore


def _cpu(read, text):
    """The least of three processor times of ``read(text)``."""
    best = None
    for _ in range(3):
        start = time.process_time()
        read(text)
        spent = time.process_time() - start
        best = spent if best is None else min(best, spent)
    return best


def _steep(read, make):
    """True when eight times the text takes far more than eight times the time, twice in a row (linear grows eight
    times, quadratic sixty-four)."""
    def once():
        small, large = _cpu(read, make(500)), _cpu(read, make(4000))
        return large > 0.002 and large > 24 * max(small, 2e-5)
    return once() and once()


# ---------------------------------------------------------------------------
# SL1 -- the verdicts of a table of patterns
# ---------------------------------------------------------------------------
_SL1_TABLE = [
    ("foo", 0, "linear"), (r"\bword\b", re.I, "linear"), (r"x+", 0, "linear"), ("", 0, "linear"),
    (r"x\s*y", 0, "linear"), (r"\A\s*\w+", 0, "linear"), (r"^\s*x", 0, "linear"), (r"(?im)^.*foo.*", 0, "linear"),
    (r"(?<=a)b", 0, "linear"), (r"/dev/(?!null|stdin)[a-z]", re.I, "linear"),
    (r"a*b", 0, "polynomial"), (r"\d+x", 0, "polynomial"), (r"(?i).*foo.*", 0, "polynomial"),
    (r"foo.*bar", 0, "polynomial"), (r"^\s*x", re.M, "polynomial"), (r"(?m)^\s*x", 0, "polynomial"),
    (r"<[^>]+>(?P<real>)|<[^>]*", 0, "polynomial"), (r"a*+b", 0, "polynomial"), (r"(?>a+)b", 0, "polynomial"),
    (r"(a+)+b", 0, "exponential"), (r"(a|a)*b", 0, "exponential"), (r"(\w+\s?)+$", 0, "exponential"),
    (r"(?=a*)b", 0, "unchecked"), (r"(a)\1", 0, "unchecked"), (r"(a)?(?(1)b|c)", 0, "unchecked"),
    ("a{0,900}", 0, "unchecked"),
]


def test_sl1_the_verdicts_of_a_table_of_patterns():
    lp, restore = _checker()
    try:
        assert {verdict for _p, _f, verdict in _SL1_TABLE} == {"linear", "polynomial", "exponential", "unchecked"}
        found = [(pattern, flags, lp.verdict(pattern, flags)) for pattern, flags, _ in _SL1_TABLE]
        assert found == _SL1_TABLE, [row for row, want in zip(found, _SL1_TABLE) if row != want]
        raised = None
        try:
            lp.verdict("(", 0)
        except re.error as exc:
            raised = exc
        assert raised is not None and not isinstance(raised, lp.CostlyPattern), "an invalid pattern is no verdict"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL2 -- soundness on a seeded corpus
# ---------------------------------------------------------------------------
_SL2_ATOMS = ["a", "b", "x", "[ab]", "[^a]", ".", r"\s", r"\w", r"\d", "[^\\n]", " ", "\\n"]
_SL2_ALPHABET = ["a", "b", "x", " ", "\n", "0"]


def _sl2_pattern(rng, depth):
    pick = rng.random()
    if depth == 0 or pick < 0.3:
        return rng.choice(_SL2_ATOMS)
    if pick < 0.55:
        return "".join(_sl2_pattern(rng, depth - 1) for _ in range(rng.randint(2, 3)))
    if pick < 0.7:
        return "(?:" + "|".join(_sl2_pattern(rng, depth - 1) for _ in range(2)) + ")"
    return "(?:" + _sl2_pattern(rng, depth - 1) + ")" + rng.choice(["*", "+", "?", "*?", "{1,3}", "{2,}"])


def _sl2_texts(pump, method):
    return lambda size: pump * (size // len(pump)) + "\x00"


def test_sl2_no_pattern_the_check_calls_linear_reads_a_long_text_in_more_than_linear_time():
    lp, restore = _checker()
    try:
        pumps = list(dict.fromkeys("".join(t) for k in (1, 2) for t in itertools.product(_SL2_ALPHABET, repeat=k)))

        def steep(compiled):
            return [(pump, method) for pump in pumps for method in ("search", "findall")
                    if _steep(getattr(compiled, method), _sl2_texts(pump, method))]

        assert steep(re.compile(r"\s*x")), "witness: the oracle sees a quadratic search"
        assert steep(re.compile(r"(?m)^\s*x")), "witness: and the starts of every line"
        swallowed = {method for _p, method in steep(re.compile(r"x[^0]+0(?P<real>)|x[^0 ]*"))}
        assert swallowed == {"findall"}, ("witness: every match in turn, where one search is linear", swallowed)
        rng, seen, linear = random.Random(7), set(), []
        while len(seen) < 120:
            pattern, flags = _sl2_pattern(rng, rng.randint(2, 3)), rng.choice([0, re.M, re.S])
            if (pattern, flags) in seen:
                continue
            seen.add((pattern, flags))
            if rng.random() < 0.2:
                pattern = "^" + pattern
            if lp.verdict(pattern, flags) == "linear":
                linear.append((pattern, flags))
        assert len(linear) > 40, ("control: the corpus holds many linear patterns", len(linear))
        slow = [(pattern, flags, found) for pattern, flags in linear for found in [steep(re.compile(pattern, flags))]
                if found]
        assert not slow, slow
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL3 -- the check's own cost
# ---------------------------------------------------------------------------
def test_sl3_the_check_s_own_cost_is_bounded():
    lp, restore = _checker()
    try:
        start = time.process_time()
        assert lp._judged("x" * 1000, 0) == ("unchecked", f"it holds an automaton of more than {lp.STATES} states")
        wide = "".join(f"[^{chr(0x100 + i)}]" for i in range(40))
        verdict, why = lp._judged(wide, 0)
        assert verdict == "unchecked" and f"more than {lp.READS} passes" in why, why
        narrow = "".join(f"[{chr(0x100 + 2 * i)}-{chr(0x103 + 2 * i)}]" for i in range(200))
        machine = lp._Automaton()
        lp._prefetch(list(lp._parser.parse(narrow, re.I)), int(re.I | re.U), machine)
        assert machine.reads == lp.READS - 1 and machine.budget < lp.STEPS - 200, "one pass, then a split it pays for"
        lp.STEPS = 40
        verdict, why = lp._judged(r"(?:ab|cd)*e", 0)
        assert verdict == "unchecked" and "more than 40 steps" in why, why
        lp.STEPS = 400_000
        assert lp.verdict(r"(?=(?:ab|cd){8})x", 0) == "linear", "256 ways through the lookahead"
        verdict, why = lp._judged(r"(?=(?:ab|cd){9})x", 0)
        assert verdict == "unchecked" and f"more than {lp.LOOKAROUND_WAYS} ways" in why, why
        deep = "(" * 3000 + "a" + ")" * 3000
        refused = None
        try:
            lp.compile_linear(deep)
        except lp.CostlyPattern as exc:
            refused = str(exc)
        assert refused and "too large" in refused, refused
        assert time.process_time() - start < 3.0, "every refusal above comes in a bounded time"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL4 -- the sets of characters are the engine's
# ---------------------------------------------------------------------------
def test_sl4_the_sets_of_characters_are_the_engine_s():
    lp, restore = _checker()
    try:
        kelvin = chr(0x212A)
        assert lp.verdict(r"^(?i:K)+k+x", 0) == "polynomial", "k is in both runs"
        assert lp.verdict(r"^(?i:K)+q+x", 0) == "linear", "control: no character in both"
        assert lp.verdict("^(?i:" + kelvin + ")+k+x", 0) == "polynomial", "the engine folds the Kelvin sign into k"
        assert lp.verdict("^" + kelvin + "+k+x", 0) == "linear", "and only with case folding"
        items = ["a", "K", "[x-z]", "s", kelvin]
        lp._RANGES.clear()
        machine = lp._Automaton()
        lp._prefetch(list(lp._parser.parse("".join(items), re.I)), int(re.I | re.U), machine)
        assert machine.reads == lp.READS - 1, "one shared pass for the narrow sets"
        shared = dict(lp._RANGES)
        assert len(shared) == len(items), shared.keys()
        for (single, flags), ranges in shared.items():
            assert lp._scan(single, flags, lp._Automaton()) == ranges, single
        assert any((0x17F, 0x17F) in ranges for ranges in shared.values()), "s holds the long s under case folding"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL5 -- compile_linear
# ---------------------------------------------------------------------------
def test_sl5_compile_linear_returns_the_pattern_or_refuses_with_a_re_error_that_says_why():
    lp, restore = _checker()
    try:
        compiled = lp.compile_linear(r"\bfoo\b", re.I)
        assert compiled.pattern == r"\bfoo\b" and compiled.flags & re.I and compiled.search("a FOO b")
        assert issubclass(lp.CostlyPattern, re.error)
        for pattern, why in ((r"(a+)+b", "exponential"), (r"foo.*bar", "quadratic"), (r"(a)\1", "does not model"),
                             ("a{99999999999}", "too large"), (b"a", "only a text pattern")):
            refused = None
            try:
                lp.compile_linear(pattern)
            except lp.CostlyPattern as exc:
                refused = str(exc)
            assert refused and why in refused, (pattern, refused)
        invalid = None
        try:
            lp.compile_linear("(")
        except re.error as exc:
            invalid = exc
        assert invalid is not None and not isinstance(invalid, lp.CostlyPattern), "invalid, refused as before"
        assert lp._judged(r"foo.*bar", 0) is lp._judged(r"foo.*bar", 0), "a verdict is kept"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL6 -- joins
# ---------------------------------------------------------------------------
_SL6_JOINS = [
    (r"\bfork\b", r"\bwhile\b.*\btrue\b", True), (r"python[23]?\s+-c\s+", r"\b(os\.|subprocess)", True),
    (r"printf\s+", r"\\x[0-9a-fA-F].*\|\s*(bash|sh|zsh)", True), (r"base64\s+(-d|--decode)", r"\|\s*bash", True),
    (r"x+", r"y", True), (r"a", r"\b", True),
    (r"foo\w*", r"bar", False), (r"a\s*", r"\s", False), (r"ab|a", r"c", False), (r"a\s+?", r"x", False),
    (r"(?s)a", r"b", False), (r"x+", r"x", False), (r"a\s+", r"\b", False),
]


def test_sl6_joins_exactly_tells_an_exact_join_from_one_that_loses_matches():
    lp, restore = _checker()
    try:
        assert [(head, tail, lp.joins_exactly(head, tail, re.I)) for head, tail, _ in _SL6_JOINS] == _SL6_JOINS
        line = "foobar"
        assert re.search(r"foo\w*.*bar", line) and not re.search("bar", line[re.match(r"foo\w*", line).end():]), \
            "witness: the first way foo\\w* ends loses the match the join has"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL7 -- the agent's grep
# ---------------------------------------------------------------------------
def test_sl7_the_agent_s_grep_refuses_a_costly_pattern_and_still_searches_a_linear_one(tmp_path):
    import importlib.util

    import test_sandbox_tools_confinement_contracts as confinement
    ctx = confinement._load()
    try:
        spec = importlib.util.spec_from_file_location("opti_oignon.linear_pattern", source("linear_pattern.py"))
        checker = importlib.util.module_from_spec(spec)
        sys.modules["opti_oignon.linear_pattern"] = checker
        spec.loader.exec_module(checker)
        (tmp_path / "notes.txt").write_text("alpha SIGNAL beta\nfoo then bar\n", encoding="ascii")
        wrapper = ctx.tools.SandboxToolSession(sandbox_mgr=confinement._Manager(workspace=str(tmp_path)),
                                               tool_registry=confinement._registry(ctx))
        wrapper.start()
        try:
            refused = wrapper.grep(r"foo.*bar", is_regex=True)
            assert refused.startswith("Error: regex pattern refused") and "is_regex=false" in refused, refused
            found = wrapper.grep(r"\bSIG\w+", is_regex=True)
            assert "notes.txt:1:" in found and "SIGNAL" in found, found
            assert "foo then bar" in wrapper.grep("foo then bar"), "control: a fixed string is searched as before"
            assert wrapper.grep("(", is_regex=True).startswith("Error: invalid regex pattern"), "invalid as before"
        finally:
            wrapper.stop()
    finally:
        sys.modules.pop("opti_oignon.linear_pattern", None)
        ctx.restore()


# ---------------------------------------------------------------------------
# SL8 -- the sanitizer's custom patterns
# ---------------------------------------------------------------------------
def test_sl8_the_sanitizer_leaves_a_costly_custom_pattern_out_and_reads_an_older_red_team_rule_in_its_line_form(
        caplog):
    from opti_oignon.rag_sanitizer import RAGSanitizer, line_form
    from opti_oignon.redteam.feedback import held_patterns
    forget = next(pattern for pattern in held_patterns() if pattern.startswith("forget"))
    custom = [{"name": "costly", "regex": r"(?:x+x+)+y"}, {"name": "kept", "regex": r"\bSIGNAL\b"},
              {"name": "older", "regex": r"(?i).*prompt_injection.*"}, {"name": "held", "regex": forget}]
    with caplog.at_level("WARNING"):
        sanitizer = RAGSanitizer({"audit": {"enabled": False}, "sanitization": {"custom_patterns": custom}})
    kept = {name: compiled.pattern for name, compiled, _w in sanitizer._custom_patterns}
    assert kept == {"kept": r"\bSIGNAL\b", "older": r"(?im)^.*prompt_injection.*", "held": forget}, kept
    assert any("costly" in record.getMessage() and "refused" in record.getMessage() for record in caplog.records)
    assert line_form(r"(?i).*a\.b.*") == r"(?im)^.*a\.b.*" and line_form(r"(?i).*a.b.*") == r"(?i).*a.b.*", \
        "only an escaped text is read in its line form"
    old, new = re.compile(r"(?i).*a\.b.*", re.I), re.compile(r"(?im)^.*a\.b.*", re.I)
    rng = random.Random(3)
    for _ in range(3000):
        text = "".join(rng.choice(["a.b", "A.B", "x", "\n", " ", "a", ".", "b"]) for _ in range(rng.randint(0, 12)))
        found = old.search(text)
        assert (found and found.span()) == ((new.search(text) or None) and new.search(text).span()), text
        assert old.sub("[x]", text) == new.sub("[x]", text), text


# ---------------------------------------------------------------------------
# SL9 -- the PII sanitizer's custom patterns
# ---------------------------------------------------------------------------
def test_sl9_the_pii_sanitizer_leaves_a_costly_custom_pattern_out(caplog):
    from opti_oignon.pii_sanitizer import PIISanitizeConfig, PIISanitizer
    with caplog.at_level("WARNING"):
        sanitizer = PIISanitizer(PIISanitizeConfig(custom_patterns=[
            {"pattern": r"(\w+\s?)+$", "replacement": "[X]"}, {"pattern": r"\bACME-\d{4}\b", "replacement": "[ID]"}]))
    assert [compiled.pattern for compiled, _r, _l in sanitizer._custom_compiled] == [r"\bACME-\d{4}\b"]
    assert any("refused" in record.getMessage() for record in caplog.records), "said, never silent"


# ---------------------------------------------------------------------------
# SL10 -- the tone shifter's custom rules
# ---------------------------------------------------------------------------
def test_sl10_the_tone_shifter_skips_a_costly_custom_rule(caplog):
    loaded, restore = isolate(targets={
        "opti_oignon.linear_pattern": source("linear_pattern.py"),
        "tone_shifter_entry_point": source("plugins", "tone-shifter", "entry_point.py")})
    try:
        shifter = loaded["tone_shifter_entry_point"]
        with caplog.at_level("WARNING"):
            out = shifter.apply_custom_rules("aaaa kind words", {r"(a+)+$": "X", r"\bkind\b": "gentle"})
        assert out == "aaaa gentle words", out
        assert any("Invalid custom regex" in record.getMessage() for record in caplog.records)
        assert shifter.apply_custom_rules("text", {}) == "text"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL11 -- the red team's rules
# ---------------------------------------------------------------------------
def test_sl11_the_red_team_writes_a_linear_fallback_rule_and_refuses_a_costly_one_before_writing(tmp_path):
    from opti_oignon.linear_pattern import verdict
    from opti_oignon.redteam import feedback
    score = types.SimpleNamespace(classification="bypass", defense_score=0.0, category="data_exfil",
                                  strategy="plain", payload_hash="h1")
    store = feedback.SuggestionStore()
    original = feedback.suggestion_store if hasattr(feedback, "suggestion_store") else None
    feedback.suggestion_store = store
    try:
        made = feedback.extract_suggestions([score])
    finally:
        if original is not None:
            feedback.suggestion_store = original
    assert [s.regex for s in made] == [r"(?im)^.*data_exfil.*"], [s.regex for s in made]
    assert verdict(made[0].regex, re.I) == "linear" and verdict(r"(?i).*data_exfil.*", re.I) == "polynomial"
    config = tmp_path / "rag.yaml"
    config.write_text("sanitization:\n  custom_patterns: []\n", encoding="utf-8")
    before = config.read_bytes()

    def accepted(regex, name):
        return feedback.Suggestion(id=name, pattern_name=name, regex=regex, source_category="c",
                                   source_strategy="s", source_payload_hash="h", confidence=1.0, status="accepted")

    assert feedback.apply_suggestion_to_config(accepted(r"(?:x+x+)+y", "costly"), config) is False
    assert config.read_bytes() == before, "refused before anything is written"
    assert feedback.apply_suggestion_to_config(accepted(r"(?i).*old_rule.*", "older"), config) is True
    forget = next(pattern for pattern in feedback.held_patterns() if pattern.startswith("forget"))
    assert verdict(forget, re.I) == "unchecked", "control: the static check cannot prove this one"
    assert feedback.apply_suggestion_to_config(accepted(forget, "held"), config) is True
    written = yaml.safe_load(config.read_text(encoding="utf-8"))["sanitization"]["custom_patterns"]
    assert written == [{"name": "older", "regex": r"(?im)^.*old_rule.*"}, {"name": "held", "regex": forget}], written


# ---------------------------------------------------------------------------
# SL12, SL13 -- the sandbox's blocked patterns
# ---------------------------------------------------------------------------
def _sandbox(tmp_path):
    """sandbox_manager.py loaded from a byte-exact copy whose configuration switches the import-time singleton off,
    with the check of patterns beside it."""
    package = tmp_path / "copy"
    (package / "config").mkdir(parents=True)
    (package / "config" / "sandbox.yaml").write_text("enabled: false\n", encoding="utf-8")
    copy = package / "sandbox_manager.py"
    copy.write_bytes(source("sandbox_manager.py").read_bytes())
    db_utils = types.ModuleType("opti_oignon.db_utils")
    db_utils.safe_connect = lambda path, *a, **k: sqlite3.connect(path)
    loaded, restore = isolate(targets={"opti_oignon.linear_pattern": source("linear_pattern.py"),
                                       "opti_oignon.sandbox_manager": copy},
                              seeded={"opti_oignon.db_utils": db_utils})
    if loaded["opti_oignon.sandbox_manager"].sandbox_manager is not None:
        restore()
        raise AssertionError("the import-time singleton must stay off in this window")
    return loaded["opti_oignon.sandbox_manager"], restore


_SL12_DEFAULTS = yaml.safe_load(source("config", "sandbox.yaml").read_text(encoding="utf-8"))["blocked_patterns"]
_SL12_OLD_RM = (r"rm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?/", r"rm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?\.\.")
_SL12_WORDS = ["fork", "while", "true", "python3", "python", "-c", "os.system", "perl", "-e", "system", "ruby",
               "base64", "-d", "--decode", "|", "bash", "sh", "printf", "\\x41", "ln", "-s", "../", "find", "/",
               "-exec", " ", "  ", "\n", "x", "rm", "-rf", "-fr", "-Rf", "..", "/dev/null", "/dev/sda", "`a`"]


def test_sl12_the_sandbox_makes_its_blocked_patterns_at_the_first_command_and_refuses_what_it_cannot_hold(tmp_path):
    sm, restore = _sandbox(tmp_path)
    try:
        assert len(_SL12_DEFAULTS) == 23, "control: the shipped defaults"
        validator = sm.CommandValidator(sm.SandboxConfig(blocked_commands=[], blocked_patterns=_SL12_DEFAULTS))
        assert validator._blocked_patterns is None, "nothing is made at construction"
        assert validator.validate("ls -la") == (True, "")
        patterns, refused = validator._blocked()
        assert refused == [] and len(patterns) == 23, refused
        chains = [p.pattern for p in patterns if isinstance(p, sm.ChainedPattern)]
        assert len(chains) == 8 and all(".*" in chain for chain in chains), chains
        rng = random.Random(11)
        commands = ["".join(rng.choice(_SL12_WORDS) + rng.choice(["", " ", " ", "\n"]) for _ in range(rng.randint(1, 9)))
                    for _ in range(6000)]
        commands += ["fork x while y true", "python3 -c 'import os; os.system(1)'", "printf '\\x41' | sh",
                     "rm -ffff /", "rm -rf ..", "perl  -e   system", "find / -name x -exec rm {} ;",
                     "ln -sf ../../etc x", "base64 -d f | bash", "python -c \n\n os.system", "cat /dev/sda",
                     "x | curl y", "a |zsh", "$( sudo x)", ":(){ :|:& };", "cat /proc/1", "ls /sys/x",
                     "export LD_PRELOAD=x", "ln -s /etc x", "echo `id`", "ruby -e system", "x | sh"]
        old_of = dict(zip([p for p in _SL12_DEFAULTS if p.startswith("rm")], _SL12_OLD_RM))
        assert len(old_of) == 2, "control: the two rm patterns, rewritten"
        hits = {}
        for pattern, prepared in zip(_SL12_DEFAULTS, patterns):
            old = re.compile(old_of.get(pattern, pattern), re.I)
            for command in commands:
                said = bool(prepared.search(command))
                assert said is bool(old.search(command)), (pattern, command)
                hits[pattern] = hits.get(pattern, 0) + said
        assert all(hits.get(pattern) for pattern in _SL12_DEFAULTS), ("witness: each pattern is reached", hits)
        refusing = sm.CommandValidator(sm.SandboxConfig(blocked_commands=[], blocked_patterns=[
            r"\bsafe\b", r"(a+)+$"]))
        assert refusing.validate("ls") == (False, "Sandbox configuration refused: the blocked pattern '(a+)+$' may "
                                                  "take more than linear time on a long command")
        inexact = sm.CommandValidator(sm.SandboxConfig(blocked_commands=[], blocked_patterns=[r"foo\w*.*bar"]))
        assert inexact.validate("ls")[0] is False and inexact._blocked_refused == [r"foo\w*.*bar"]
        chained = sm.CommandValidator(sm.SandboxConfig(blocked_commands=[], blocked_patterns=[r"curl.*\|\s*sh", "("]))
        assert chained.validate("curl x | sh") == (False, r"Blocked pattern: curl.*\|\s*sh")
        assert chained.validate("curl x") == (True, ""), "an invalid pattern is left out with a warning, as before"
    finally:
        restore()


def test_sl13_a_chain_reads_a_long_command_in_linear_time(tmp_path):
    sm, restore = _sandbox(tmp_path)
    try:
        validator = sm.CommandValidator(sm.SandboxConfig(blocked_commands=[], blocked_patterns=_SL12_DEFAULTS))
        patterns, _refused = validator._blocked()
        fork = next(p for p in patterns if p.pattern.startswith(r"\bfork\b"))
        printf = next(p for p in patterns if p.pattern.startswith("printf"))
        for unit in ("fork ", "fork while ", "while true fork "):
            assert not _steep(fork.search, lambda size: unit * (size // len(unit))), unit
        assert not _steep(printf.search, lambda size: "printf \\x41 " * (size // 12)), "printf"
        old = re.compile(fork.pattern, re.I)
        assert _steep(old.search, lambda size: "fork while " * (size // 11)), "witness: the joined pattern was not"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL14 -- the RAG store's page cleaning
# ---------------------------------------------------------------------------
def _old_tags(text, tag):
    text = re.sub(rf"<{tag}[\s>].*?</{tag}>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(rf"<{tag}\s[^>]*/?>", " ", text, flags=re.IGNORECASE)


def _old_boilerplate(text, word):
    return re.sub(rf'<(?:div|section|aside|ul|ol)[^>]*(?:class|id)=["\'][^"\']*{re.escape(word)}[^"\']*["\'][^>]*>'
                  rf'.*?</(?:div|section|aside|ul|ol)>', " ", text, flags=re.DOTALL | re.IGNORECASE)


_SL14_PINNED = ['<div class="x" id="x></div>">rest</div>tail', '<div class="a" id="x">in</div>out',
                '<ul data-class="xx">a</ul>', "<div id='x'>a</ul>b</div>", '<div class="x>y">a</div>',
                '<DIV CLASS="X">a</div>', '<div class="x"', '<div class="x">no closing']


def _sl14_tag(rng):
    out = "<" + rng.choice(["div", "UL", "section", "ol", "aside", "span"])
    for _ in range(rng.randint(0, 3)):
        quote = rng.choice(['"', "'"])
        out += rng.choice([" ", "", "\n"]) + rng.choice(["class=", "id=", "data-class=", "x="]) + quote
        out += "".join(rng.choice(["x", "X", "a", ">", "</div>", " ", ""]) for _ in range(rng.randint(0, 3)))
        out += rng.choice([quote, '"', "'", ""])
    return out + rng.choice([">", "", " >"])


def test_sl14_the_rag_store_cleans_a_page_as_the_patterns_it_replaced_did_in_linear_time():
    loaded, restore = isolate(targets={"opti_oignon.rag_store": source("rag_store.py")})
    try:
        store = loaded["opti_oignon.rag_store"]
        strip, extract = store._strip_boilerplate, store.RAGVectorStore._extract_html_text
        for case in _SL14_PINNED:
            assert strip(case, "x") == _old_boilerplate(case, "x"), case
        assert strip(_SL14_PINNED[0], "x") == " tail", "the last attribute before the first > decides"
        for size in range(6):
            for parts in itertools.product(["<div", "</div>", " class=", " id=", '"', "'", "x", ">", " "],
                                           repeat=size):
                page = "".join(parts)
                assert strip(page, "x") == _old_boilerplate(page, "x"), page
                assert strip(page, "") == _old_boilerplate(page, ""), page
        rng = random.Random(13)
        for _ in range(4000):
            page = "".join(_sl14_tag(rng) if rng.random() < 0.6 else rng.choice(["</div>", "</ul>", "text", ">", "<"])
                           for _ in range(rng.randint(1, 8)))
            assert strip(page, "x") == _old_boilerplate(page, "x"), page
        for size in range(7):
            for parts in itertools.product(["<a", "</a>", ">", " ", "x", "\n", "<A"], repeat=size):
                page = "".join(parts)
                cfg = {"strip_tags": ["a"], "boilerplate_patterns": []}
                assert extract(page, cfg) == extract(_old_tags(page, "a"), {"strip_tags": [], "boilerplate_patterns": []}), \
                    page
        assert extract("<a+b>x</a+b> <a>y</a>", {"strip_tags": ["a+b"], "boilerplate_patterns": []}) == "y", \
            "a tag's name is read as written"
        cfg = {"strip_tags": ["script"], "boilerplate_patterns": ["menu"]}
        for unit in ("<script ", '<div class="menu" ', "<div class=' ", "<svg "):
            assert not _steep(lambda page: extract(page, dict(cfg, strip_tags=["script", "svg"])),
                              lambda size: unit * (size // len(unit))), unit
        assert _steep(lambda page: _old_tags(page, "script"), lambda size: "<script " * (size // 8)), \
            "witness: the old pattern was not linear"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL15 -- a bound address and a section header
# ---------------------------------------------------------------------------
def test_sl15_a_bound_address_is_read_as_before_and_a_section_header_within_its_line():
    guard_source = source("network_bind_guard.py").read_text(encoding="utf-8")
    assert '(?<![\\d.*:])([\\d.*:]+):{port}\\b' in guard_source, "the pattern the guard reads ss with"
    new = re.compile(r"(?<![\d.*:])([\d.*:]+):80\b")
    old = re.compile(r"([\d.*:]+):80\b")
    for size in range(7):
        for parts in itertools.product(["1", ".", ":", "*", ":80", "a", " ", "8", "0"], repeat=size):
            line = "".join(parts)
            found, was = new.search(line), old.search(line)
            assert (found and (found.span(), found.group(1))) == (was and (was.span(), was.group(1))), line
    assert not _steep(new.search, lambda size: ":" * size) and _steep(old.search, lambda size: ":" * size), \
        "linear now, and the witness: it was not"
    skills_source = source("agent", "skills.py").read_text(encoding="utf-8")
    start = skills_source.index("def _extract_section")
    namespace = {"re": re}
    exec(skills_source[start:skills_source.index("\n\n\n", start)], namespace)
    extract = namespace["_extract_section"]
    body = "intro\n## Verification:\nrun the tests\n## Next\nmore\n"
    assert extract(body, "Verification") == "run the tests"
    assert extract("Verification  \nline\n", "verification") == "line", "blanks, any case"
    assert extract("#\n\nVerification\nline\n# Other\n", "Verification") == "line", "a line of its own"
    assert extract("Verification x\n", "Verification") == "", "a header holds nothing else"
    assert not _steep(lambda text: extract(text, "Verification"), lambda size: "\n" * size), "blank lines"
    assert not _steep(lambda text: extract(text, "Verification"), lambda size: "Verification" + " " * size + "x")
    assert _steep(lambda text: re.search(r"^#{0,6}\s*Verification\s*:?\s*$", text, re.I | re.M),
                  lambda size: "\n" * size), "witness: the old header pattern was not"


# ---------------------------------------------------------------------------
# SL16 -- a line start under MULTILINE, after a line of any characters
# ---------------------------------------------------------------------------
# Each line holds a character other than a line break, so a reader of line starts that only came back to a line
# start after line breaks would see one start; the engine tries every one.
_SL16_LINES = [(r"^(?:a\n)*x", re.M, "a\n"), (r"(?m)^(?:\w\n)+y", 0, "b\n"), (r"^(?:ab\n)*x", re.M, "ab\n")]


def test_sl16_a_line_start_under_multiline_is_read_after_every_line_not_only_after_blank_ones():
    lp, restore = _checker()
    try:
        for pattern, flags, line in _SL16_LINES:
            compiled = re.compile(pattern, flags)
            assert _steep(compiled.search, lambda size: line * (size // len(line))), ("witness: quadratic", pattern)
            assert lp.verdict(pattern, flags) == "polynomial", pattern
        assert lp.verdict(r"^(?:a\n)*x", 0) == "linear", "control: without MULTILINE one start, read once"
        assert not _steep(re.compile(r"^(?:a\n)*x").search, lambda size: "a\n" * (size // 2)), "as the engine reads it"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL17 -- a head that may end where another way reads on does not join exactly
# ---------------------------------------------------------------------------
# Each head matches a text that is a proper prefix of another it matches, read by two ways of its automaton: one
# ends where the other still reads ("a" and "aa" for a?a), so the first way a reader takes may end too early.
_SL17_HEADS = [("a?a", "b"), (r"x?\w", "a"), (r"\d\d?\d", r"\d"), ("[ab]{1,2}a", "c"), (r"rm\s+-r?\w", "f")]
_SL17_ATOMS = ["a", "b", " ", r"\s", r"\w", "[ab]", "[^a]", r"\d", "-", "/", "x", r"\n"]
_SL17_COUNTS = ["", "", "", "*", "+", "?", "{1,2}", "*?", "+?", "{2,}"]


def _sl17_part(rng, depth=0):
    out = []
    for _ in range(rng.randint(1, 3)):
        if rng.random() < 0.15 and depth < 2:
            out.append("(?:" + _sl17_part(rng, depth + 1) + "|" + _sl17_part(rng, depth + 1) + ")"
                       + rng.choice(_SL17_COUNTS))
        else:
            out.append(rng.choice(_SL17_ATOMS) + rng.choice(_SL17_COUNTS))
    return "".join(out)


def test_sl17_a_head_that_may_end_where_another_way_reads_on_does_not_join_exactly(tmp_path):
    import importlib
    sm, restore = _sandbox(tmp_path)
    try:
        lp = importlib.import_module("opti_oignon.linear_pattern")
        for head, tail in _SL17_HEADS:
            assert lp.joins_exactly(head, tail, re.I) is False, head
        assert lp.joins_exactly(r"\bsudo\s+", "rm", re.I) and lp.joins_exactly(r"foo\w*", "-", re.I), \
            "control: heads that end one way only, or with a run the tail cannot start"
        command = "rm -rf"
        assert re.search(r"rm\s+-r?\w.*f", command, re.I) and not re.search(
            "f", command[re.match(r"rm\s+-r?\w", command).end():]), "witness: the first way loses the match"
        custom = sm.CommandValidator(sm.SandboxConfig(blocked_commands=[], blocked_patterns=[r"rm\s+-r?\w.*f"]))
        assert custom.validate(command)[0] is False and custom._blocked_refused == [r"rm\s+-r?\w.*f"], \
            "a custom rule that cannot be chained exactly refuses, never stops blocking"
        rng = random.Random(1)
        exact = lost = 0
        while exact < 300:
            joined = ".*".join(_sl17_part(rng) for _ in range(rng.choice([2, 2, 3])))
            try:
                whole = re.compile(joined, re.I)
            except re.error:
                continue
            parts = sm._joined_parts(joined)
            if parts is None or not all(parts):
                continue
            try:
                chain = sm.ChainedPattern(joined, [lp.compile_linear(part, whole.flags) for part in parts])
            except re.error:
                continue
            texts = ["".join(rng.choice("aAbx -/1\n") for _ in range(rng.randint(0, 14))) for _ in range(60)]
            differs = [text for text in texts if (chain.search(text) is None) != (whole.search(text) is None)]
            if all(lp.joins_exactly(parts[at], ".*".join(parts[at + 1:]), whole.flags)
                   for at in range(len(parts) - 1)):
                exact += 1
                assert not differs, (joined, differs[:3])
            else:
                lost += bool(differs)
        assert lost, "witness: the oracle sees chains that lose matches where a join is not exact"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL18 -- a group's type flag takes the place of the pattern's
# ---------------------------------------------------------------------------
def test_sl18_a_group_s_type_flag_takes_the_place_of_the_pattern_s_as_in_the_engine():
    lp, restore = _checker()
    try:
        accent = chr(0xE9)
        assert [lp.verdict(pattern, flags) for pattern, flags in [
            (r"(?a:\w)+", 0), (r"(?a:k)x", re.I), (r"x(?u:\w)", re.A), (r"(?a:\s)", 0), (r"[a-z]+(?a:\d)", 0)]] == [
            "linear", "linear", "linear", "linear", "polynomial"]
        assert lp.compile_linear(r";(?a:\w)+;").pattern == r";(?a:\w)+;"
        assert lp.joins_exactly(r"(?a:\w)", "b", re.I | re.U) and lp.joins_exactly("a", r"(?a:\d)", re.I | re.U)
        run = "-" + accent + "*"
        assert lp.verdict(run + r"(?a:\w)*y") == "linear", "ASCII words inside the group: a run of accents splits once"
        assert lp.verdict(run + r"\w*y") == "polynomial", "control: Unicode words share the accent"
        assert lp.verdict(run + r"(?u:\w)*y", re.A) == "polynomial", "Unicode words inside the group, under ASCII"
        assert lp.verdict(run + r"\w*y", re.A) == "linear", "control: ASCII words outside"
        assert _steep(re.compile(run + r"(?u:\w)*y", re.A).search, lambda size: "-" + accent * size) and not _steep(
            re.compile(run + r"(?a:\w)*y").search, lambda size: "-" + accent * size), "witness: as the engine reads"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL19 -- a check holds the sets it read, whatever another check empties
# ---------------------------------------------------------------------------
def test_sl19_a_check_holds_the_sets_it_read_whatever_another_check_empties():
    lp, restore = _checker()
    try:
        scans, engine_pass, keep = [], lp._scan, lp._keep

        def counted(single, flags):
            scans.append(single)
            return engine_pass(single, flags)

        lp._RANGES.clear()
        lp._scan, lp._keep = counted, lambda key, ranges: lp._RANGES.clear()
        assert lp.verdict(r";\w;") == "linear"
        assert scans == [r"[\w]"], ("read once, held though the kept sets were emptied at once", scans)
        lp._scan, lp._keep = engine_pass, keep
        lp._RANGES.clear()
        lp._RANGES.update({("filler%d" % index, 0): ((0, 0),) for index in range(lp.KEPT - 32)})
        word = "abcdefghijklmnopqrstuvwxyz0123456789"
        assert lp.verdict(word, re.I) == "linear", "a cache near its bound, emptied while the check reads its sets"
        assert lp.compile_linear(word, re.I).pattern == word
        assert len(lp._RANGES) < lp.KEPT - 32, "witness: the kept sets were emptied during the check"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL20 -- an older red-team rule whose text holds a line break keeps its written form
# ---------------------------------------------------------------------------
def test_sl20_an_older_red_team_rule_whose_text_holds_a_line_break_keeps_its_written_form(caplog):
    from opti_oignon.rag_sanitizer import RAGSanitizer, line_form
    for text in ("\nfoo", "a\nb", "foo\n"):
        written = "(?i).*" + re.escape(text) + ".*"
        assert line_form(written) == written, text
    assert line_form("(?i).*" + re.escape("a b") + ".*") == r"(?im)^.*a\ b.*", "control: one line, its line form"
    page = "a\nfoo\nfoo"
    old, lined = re.compile(r"(?i).*\nfoo.*"), re.compile(r"(?im)^.*\nfoo.*")
    assert old.findall(page) == ["a\nfoo", "\nfoo"] and lined.findall(page) == ["a\nfoo"], \
        "witness: the line form loses the match that starts at the end of the last one"
    with caplog.at_level("WARNING"):
        sanitizer = RAGSanitizer({"audit": {"enabled": False}, "sanitization": {"custom_patterns": [
            {"name": "broken", "regex": "(?i).*" + re.escape("\nfoo") + ".*"}]}})
    assert sanitizer._custom_patterns == [], "a costly rule as written is left out"
    assert any("broken" in record.getMessage() and "refused" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# SL21 -- a colon alone past the header's line still closes the header
# ---------------------------------------------------------------------------
def _sl21_old_extract(body, header):
    found = re.compile(r"^#{0,6}\s*" + re.escape(header) + r"\s*:?\s*$", re.IGNORECASE | re.MULTILINE).search(body)
    if not found:
        return ""
    after = re.compile(r"^#{1,6}\s+\S", re.MULTILINE).search(body, found.end())
    return body[found.end():after.start() if after else len(body)].strip()


def test_sl21_a_colon_alone_past_the_header_s_line_still_closes_the_header_as_before():
    skills_source = source("agent", "skills.py").read_text(encoding="utf-8")
    start = skills_source.index("def _extract_section")
    namespace = {"re": re}
    exec(skills_source[start:skills_source.index("\n\n\n", start)], namespace)
    extract = namespace["_extract_section"]
    assert extract("steps\n:\nrun the tests\n", "steps") == "run the tests"
    assert extract("steps \r\n\n  :  \nrun\n", "Steps") == "run", "past blank lines and blanks"
    assert extract("steps\n:x\n", "steps") == ":x", "control: a colon that leads a line of text is text"
    assert extract("steps:\n:\nrun\n", "steps") == ":\nrun", "control: one colon closes the header"
    assert extract("###### steps\nrun\n", "steps") == _sl21_old_extract("###### steps\nrun\n", "steps") == "run"
    assert extract("####### steps\nrun\n", "steps") == _sl21_old_extract("####### steps\nrun\n", "steps") == "", \
        "six marks at most, as before"
    pieces = ["steps", ":", "\n", " ", "#", "x", "\r", "# N\n", ":x"]
    seen = 0
    for size in range(6):
        for parts in itertools.product(pieces, repeat=size):
            body = "".join(parts)
            assert extract(body, "steps") == _sl21_old_extract(body, "steps"), body
            seen += body.count(":") > 0 and "steps" in body
    assert seen > 1000, "witness: the enumeration holds headers with colons"
    assert not _steep(lambda text: extract(text, "steps"), lambda size: "steps" + "\n" * size + "x")


# ---------------------------------------------------------------------------
# SL22 -- a blocked pattern is cut at its top level only
# ---------------------------------------------------------------------------
_SL22_CUTS = [
    (r"a.*b", ["a", "b"]), (r"a.*?b", ["a", "b"]), (r"a.*b.*?c", ["a", "b", "c"]), (r"a\.*b", None),
    (r"a\\.*b", ["a\\\\", "b"]), (r"[.*]b", None), (r"a[].*]b", None), (r"a[^].*]b", None), (r"a[\].*]b", None),
    (r"a[]].*b", ["a[]]", "b"]), (r"a[^]].*b", ["a[^]]", "b"]), (r"(a.*b)c", None),
    (r"(?:a.*b).*c", ["(?:a.*b)", "c"]), (r"a(?#.*).*b", ["a(?#.*)", "b"]), (r".*a", ["", "a"]),
    (r"a.*.*b", ["a", "", "b"]),
]
_SL22_ATOMS = ["a", "b", r"\.", r"\\", "[.*]", "[]a]", "[^]a]", r"[\]b]", "(?:a.*b)", "(?#.*)", r"\*", " ", r"\s", "x+"]


def test_sl22_a_blocked_pattern_is_cut_at_its_top_level_only(tmp_path):
    sm, restore = _sandbox(tmp_path)
    try:
        import importlib
        lp = importlib.import_module("opti_oignon.linear_pattern")
        assert [(pattern, sm._joined_parts(pattern)) for pattern, _ in _SL22_CUTS] == _SL22_CUTS
        rng = random.Random(5)
        accepted = 0
        while accepted < 300:
            parts = ["".join(rng.choice(_SL22_ATOMS) for _ in range(rng.randint(1, 3)))
                     for _ in range(rng.choice([2, 2, 3]))]
            joined = parts[0] + "".join(rng.choice([".*", ".*?"]) + part for part in parts[1:])
            flags = re.compile(joined, re.I).flags
            assert sm._joined_parts(joined) == parts, joined
            if not all(lp.joins_exactly(parts[at], ".*".join(parts[at + 1:]), flags) for at in range(len(parts) - 1)):
                continue
            try:
                chain = sm.ChainedPattern(joined, [lp.compile_linear(part, flags) for part in parts])
            except re.error:
                continue
            accepted += 1
            for _ in range(80):
                text = "".join(rng.choice("ab.*]\\ \nx") for _ in range(rng.randint(0, 12)))
                assert (chain.search(text) is None) is (re.search(joined, text, re.I) is None), (joined, text)
        wrong = sm.ChainedPattern(r"a.*?[bc]d", [re.compile("a"), re.compile("bc]d")])
        assert re.search(r"a.*?[bc]d", "a bd") and wrong.search("a bd") is None, \
            "witness: a part cut one character late loses the match"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL23 -- a range is read whole
# ---------------------------------------------------------------------------
def test_sl23_a_range_of_a_class_is_read_whole():
    lp, restore = _checker()
    try:
        assert lp.verdict(r"[a-z]x*y") == "polynomial", "x is in the range past its first member"
        assert lp.verdict(r"[a-w]x*y") == "linear", "control: x is past the range"
        assert _steep(re.compile(r"[a-z]x*y").search, lambda size: "x" * size), "witness: quadratic on a run of x"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL24 -- a head that ends with a run joins only when nothing is lost
# ---------------------------------------------------------------------------
# Each join loses a match when the reader takes the first way the head matches: the text is the witness. The bounded
# repeat x{2} gives the head's part more states than items, so the tail's first state is not numbered by the items.
_SL24_LOST = [(r"a(?:bc)?b+", "c", "abcb"), (r"x{2}a+", r"(?<=xa)", "xxaa"), (r"x{2}a+", "a", "xxaa"),
              (r"x{2}a+", "[ab]", "xxaa")]


def test_sl24_a_head_that_ends_with_a_run_joins_only_when_nothing_is_lost():
    lp, restore = _checker()
    try:
        for head, tail, text in _SL24_LOST:
            first = re.match(head, text, re.I)
            assert re.search(head + ".*" + tail, text, re.I) and re.compile(tail, re.I).search(
                text, first.end()) is None, ("witness: the first way loses the match", head, tail)
            assert lp.joins_exactly(head, tail, re.I) is False, (head, tail)
        assert lp.joins_exactly(r"x{2}a+", "b", re.I) is True, "control: a part that ends once, a tail the run refuses"
        assert lp.joins_exactly(r"\b", "x", re.I) is True and lp.joins_exactly("^", "x", re.I) is True, \
            "a head that reads no character ends once"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL25 -- a chain tries every start of its head and every place of its tail to its line's break
# ---------------------------------------------------------------------------
# (pattern, text): the head's first start has no tail on its line, the next start one place on does; the tail
# matches the line break itself; a head that matches empty at the end and no tail.
_SL25_CASES = [(r"\s.*x", "\n\nx"), (r"a.*\s", "a\n"), (r"a.*\s", "ab\n"), (r"(?#.*).* ", ""), (r"(?#.*).* ", "x"),
               (r"\s.*x", "\n\ny")]


def test_sl25_a_chain_tries_every_start_of_its_head_and_every_place_of_its_tail_to_its_line_break(tmp_path):
    sm, restore = _sandbox(tmp_path)
    try:
        for pattern, text in _SL25_CASES:
            validator = sm.CommandValidator(sm.SandboxConfig(blocked_commands=[], blocked_patterns=[pattern]))
            patterns, refused = validator._blocked()
            assert refused == [] and isinstance(patterns[0], sm.ChainedPattern), ("control: read as a chain", pattern)
            assert bool(patterns[0].search(text)) is bool(re.search(pattern, text, re.I)), (pattern, text)
        assert [bool(re.search(pattern, text, re.I)) for pattern, text in _SL25_CASES] == [
            True, True, True, False, False, False], "witness: the cases hold matches and misses"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL26 -- every refusal of the check is a re.error, every doubt of a join is False
# ---------------------------------------------------------------------------
# The engine refuses these with an error that is not a re.error: a repeat count too large, flags that cannot stand
# together.
_SL26_ODD = [("a{4294967296}", OverflowError), ("(?a)(?u)x", ValueError)]


def test_sl26_every_refusal_of_the_check_is_a_re_error_and_every_doubt_of_a_join_is_false(tmp_path):
    sm, restore = _sandbox(tmp_path)
    try:
        import importlib
        lp = importlib.import_module("opti_oignon.linear_pattern")
        for pattern, error in _SL26_ODD:
            try:
                re.compile(pattern)
            except error as exc:
                assert not isinstance(exc, re.error), "witness: the engine's own error is not a re.error"
            else:
                raise AssertionError(("witness: the engine refuses it", pattern))
            for check in (lp.verdict, lp.compile_linear):
                try:
                    check(pattern)
                except re.error:
                    pass
                else:
                    raise AssertionError((check.__name__, pattern))
            assert lp.joins_exactly(pattern, "x") is False and lp.joins_exactly("x", pattern) is False, pattern
        large = sm.CommandValidator(sm.SandboxConfig(blocked_commands=[], blocked_patterns=["a{4294967296}"]))
        for _ in range(2):
            assert large.validate("ls") == (False, "Sandbox configuration refused: the blocked pattern "
                                                   "'a{4294967296}' may take more than linear time on a long command")
        flags = sm.CommandValidator(sm.SandboxConfig(blocked_commands=[], blocked_patterns=["(?a)(?u)x", "rm"]))
        assert flags.validate("ls") == (True, "") and flags.validate("rm x")[0] is False, \
            "flags that cannot stand together make an invalid pattern, left out with a warning, as before"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL27 -- a repeat counted past the bound, or a lookaround longer than it, is refused at once
# ---------------------------------------------------------------------------
_SL27_HUGE = ["(?:){0,4294967294}", "(?:){4000000}", "x(?=a{4000000})", "x(?=(?:ab){0,4000000})",
              "(?:(?:){600}){600}", "(?=a{5000})"]


def test_sl27_a_repeat_counted_past_the_bound_or_a_lookaround_longer_than_it_is_refused_at_once():
    lp, restore = _checker()
    try:
        start = time.process_time()
        for pattern in _SL27_HUGE:
            assert lp.verdict(pattern) == "unchecked", pattern
        assert time.process_time() - start < 2.0, "each refusal comes before the repeat is unrolled"
        assert lp.verdict("(?=a{500})") == "linear", "control: a lookaround within the bound is an assertion"
        long, short = re.compile("(?=a{%d})" % (4 * lp.STATES)), re.compile("(?=a{8})")
        assert _cpu(long.findall, "a" * 8000) > 10 * _cpu(short.findall, "a" * 8000), \
            "witness: a lookaround costs its length wherever it is tried, whatever its ways"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL28 -- a verdict does not hang on what earlier checks kept
# ---------------------------------------------------------------------------
def test_sl28_a_verdict_does_not_hang_on_what_earlier_checks_kept():
    lp, restore = _checker()
    try:
        classes = ["[^%s]" % letter for letter in "abcdefghijklmnopqrstuvwxyzABCDEFGH"]
        pattern = "".join(classes)
        lp._RANGES.clear()
        cold = lp.verdict(pattern)
        lp._RANGES.clear()
        assert lp.verdict("".join(classes[:20])) == "linear"
        assert len(lp._RANGES) >= 20, "witness: an earlier check kept twenty of its sets"
        warm = lp.verdict(pattern, re.M)
        assert cold == warm == "unchecked", ("multiline changes no set: the same sets, judged again", cold, warm)
        blocks = ["[%s-%s]" % (chr(0x4E00 + 0x1000 * index), chr(0x4E00 + 0x1000 * index + 0xFFF)) for index in range(33)]
        lp._RANGES.clear()
        cold = lp.verdict("".join(blocks))
        lp._RANGES.clear()
        assert lp.verdict("".join(blocks[:20])) == "linear", "twenty narrow sets read and kept"
        warm = lp.verdict("".join(blocks), re.M)
        assert cold == warm == "unchecked", ("narrow sets past FEW: one shared pass and one each, kept or not", cold, warm)
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL29 -- a pattern or flags of the wrong type are refused as invalid, never escape
# ---------------------------------------------------------------------------
def test_sl29_a_pattern_or_flags_of_the_wrong_type_are_refused_as_invalid_and_never_escape(tmp_path):
    sm, restore = _sandbox(tmp_path)
    try:
        import importlib
        lp = importlib.import_module("opti_oignon.linear_pattern")
        for pattern, flags in ((123, 0), (["x"], 0), ("a", "x"), ("a", None)):
            for check in (lp.verdict, lp.compile_linear):
                try:
                    check(pattern, flags)
                except lp.CostlyPattern as exc:
                    raise AssertionError(("invalid, not costly", check.__name__, pattern, flags)) from exc
                except re.error:
                    pass
                else:
                    raise AssertionError((check.__name__, pattern, flags))
        assert lp.joins_exactly(123, "x") is False and lp.joins_exactly("x", ["y"]) is False
        assert lp.joins_exactly("a", "x", 1.5) is False and lp.joins_exactly("a", "x", "i") is False, \
            "flags of the wrong type are doubted, never truncated"
        number = sm.CommandValidator(sm.SandboxConfig(blocked_commands=[], blocked_patterns=[123, "rm"]))
        assert number.validate("ls") == (True, "") and number.validate("rm x")[0] is False, \
            "a rule that is not text is left out with a warning, as an invalid one"
        raw = sm.CommandValidator(sm.SandboxConfig(blocked_commands=[], blocked_patterns=[b"rm.*-rf"]))
        for _ in range(2):
            assert raw.validate("ls")[0] is False and raw._blocked_refused == [b"rm.*-rf"], \
                "a rule of bytes cannot read a command: every command is refused, naming it"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL30 -- a lookaround nested in another multiplies its ways
# ---------------------------------------------------------------------------
def test_sl30_a_lookaround_nested_in_another_multiplies_its_ways():
    lp, restore = _checker()
    try:
        assert lp.verdict(r"(?=(?:a|aa){8}(?=(?:a|aa){8}!)!)") == "unchecked", "256 ways tried on each of 256"
        assert lp.verdict(r"(?=(?:a|aa){8}(?=(?:a|aa){8}(?=(?:a|aa){8}!)!)!)") == "unchecked"
        assert lp.verdict(r"(?=(?:a|b){4}(?=(?:a|b){4}))x") == "linear", "control: 16 ways on each of 16"
        assert lp.verdict(r"(?=(?:a|aa){3}(?=(?:a|aa){6}))x") == "unchecked", "8 ways on each of 64: 512"
        assert lp.verdict(r"(?=(?:a|aa){3}(?=(?:a|aa){4}))x") == "linear", "control: 8 ways on each of 16: 128"
        assert lp.verdict(r"(?=(?:a|aa){8})x") == "linear", "control: one lookaround of 256 ways (16 steps each)"
        depth1, depth2 = re.compile(r"(?=(?:a|aa){8}!)"), re.compile(r"(?=(?:a|aa){8}(?=(?:a|aa){8}!)!)")
        assert _cpu(depth2.findall, "a" * 64) > 30 * _cpu(depth1.findall, "a" * 64), \
            "witness: the inner lookaround is tried on each way through the outer one"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL31 -- sets written too long are refused before they are read
# ---------------------------------------------------------------------------
def test_sl31_sets_of_characters_written_too_long_are_refused_before_they_are_read():
    lp, restore = _checker()
    try:
        wide = "[" + "".join(chr(0x10000 + 2 * index) for index in range(16000)) + "]"
        many = "".join("[" + "".join(chr(0x20000 + 40 * row + index) for index in range(20)) + "]" for row in range(200))
        start = time.process_time()
        for pattern in (wide, many):
            verdict, why = lp._judged(pattern, 0)
            assert verdict == "unchecked" and f"more than {lp.SOURCES} characters" in why, why
        assert time.process_time() - start < 1.0, "refused before a pass over every code point is made"
        assert lp.verdict("[" + "".join(chr(0x10000 + 2 * index) for index in range(8)) + "]x") == "linear", "control"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL32 -- a lookaround is judged once for a check, however often it is built
# ---------------------------------------------------------------------------
def test_sl32_a_lookaround_is_judged_once_for_a_check_however_often_it_is_built():
    lp, restore = _checker()
    try:
        pattern = "(?:(?=" + "(?=a)" * 3 + ")){590}"
        start = time.process_time()
        assert lp.verdict(pattern) == "linear"
        assert time.process_time() - start < 0.5, "the body is judged once, not once for each of 590 rounds"
        judged = []
        ways = lp._ways

        def counted(seq):
            judged.append(1)
            return ways(seq)

        lp._ways = counted
        assert lp.verdict("(?:(?=a)){7}b") == "linear"
        assert len(judged) == 1, ("witness: one judgement for seven rounds", len(judged))
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL33 -- a lookaround is bounded by the steps a way through it takes, rounds of zero-width items included
# ---------------------------------------------------------------------------
_SL33_ROUNDS = [r"(?=(?:(?=a)){100000})a", r"(?=(?:(?:(?=a)){600}){600})a", r"(?=(?:\b){4294967294})a",
                r"(?=(?:){4294967294})a"]


def test_sl33_a_lookaround_is_bounded_by_the_steps_a_way_through_it_takes_zero_width_rounds_included():
    lp, restore = _checker()
    try:
        start = time.process_time()
        for pattern in _SL33_ROUNDS:
            assert lp.verdict(pattern) == "unchecked", pattern
        assert time.process_time() - start < 1.0
        assert lp.verdict(r"(?=(?:(?=a)){100})a") == "linear", "control: a hundred rounds"
        assert lp.verdict(r"(?=(?:(?=a)){%d})a" % (lp.STATES // 2)) == "linear", "at the bound: two steps a round"
        assert lp.verdict(r"(?=(?:(?=a)){%d})a" % (lp.STATES // 2 + 1)) == "unchecked", "one round past it"
        assert lp.verdict(r"(?=(?:\b){%d})a" % lp.STATES) == "linear" and lp.verdict(
            r"(?=(?:\b){%d})a" % (lp.STATES + 1)) == "unchecked", "an anchor is one step"
        many, few = re.compile(r"(?=(?:(?=a)){2000})a"), re.compile(r"(?=(?:(?=a)){20})a")
        assert _cpu(many.findall, "a" * 2000) > 20 * _cpu(few.findall, "a" * 2000), \
            "witness: the engine takes every round of a zero-width repeat on every try"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL34 -- the tone shifter applies no custom rule where the check cannot be loaded
# ---------------------------------------------------------------------------
def test_sl34_the_tone_shifter_applies_no_custom_rule_where_the_check_cannot_be_loaded(caplog):
    loaded, restore = isolate(targets={"tone_shifter_entry_point": source("plugins", "tone-shifter", "entry_point.py")},
                              blocked=("opti_oignon.linear_pattern",))
    try:
        shifter = loaded["tone_shifter_entry_point"]
        with caplog.at_level("WARNING"):
            out = shifter.apply_custom_rules("kind words", {r"\bkind\b": "gentle"})
        assert out == "kind words", out
        assert any("pattern cost check is not available" in record.getMessage() for record in caplog.records)
        assert shifter.apply_custom_rules("text", {}) == "text"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL35 -- a blocked pattern whose .* crosses lines is never read as a chain of lines
# ---------------------------------------------------------------------------
def test_sl35_a_blocked_pattern_whose_join_crosses_lines_is_never_read_as_a_chain_of_lines(tmp_path):
    sm, restore = _sandbox(tmp_path)
    try:
        for pattern in (r"(?s)curl.*\|\s*sh", r"curl(?s:.*)\|\s*sh"):
            assert re.search(pattern, "curl x\n| sh"), ("witness: it matches across a line break", pattern)
        import importlib
        lp = importlib.import_module("opti_oignon.linear_pattern")
        assert lp.joins_exactly("curl", r"\|\s*sh", re.S) is False, "the check refuses a join across lines"
        dotall = sm.CommandValidator(sm.SandboxConfig(blocked_commands=[], blocked_patterns=[r"(?s)curl.*\|\s*sh"]))
        assert dotall.validate("ls")[0] is False and dotall._blocked_refused == [r"(?s)curl.*\|\s*sh"], \
            "refused, naming it: a chain reads each part to the end of its line"
        joins, lp.joins_exactly = lp.joins_exactly, lambda *args: True
        try:
            alone = sm.CommandValidator(sm.SandboxConfig(blocked_commands=[], blocked_patterns=[r"(?s)curl.*\|\s*sh"]))
            assert alone.validate("ls")[0] is False, "the sandbox refuses it too, whatever the check says"
        finally:
            lp.joins_exactly = joins
        lines = sm.CommandValidator(sm.SandboxConfig(blocked_commands=[], blocked_patterns=[r"curl.*\|\s*sh"]))
        assert lines.validate("curl x | sh")[0] is False and lines.validate("ls") == (True, ""), "control: as a chain"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL36 -- a stripped tag is read as before after an equals sign, a slash or a tab
# ---------------------------------------------------------------------------
def test_sl36_a_stripped_tag_is_read_as_before_after_an_equals_sign_a_slash_or_a_tab():
    loaded, restore = isolate(targets={"opti_oignon.rag_store": source("rag_store.py")})
    try:
        extract = loaded["opti_oignon.rag_store"].RAGVectorStore._extract_html_text
        cfg, bare = {"strip_tags": ["a"], "boilerplate_patterns": []}, {"strip_tags": [], "boilerplate_patterns": []}
        seen = 0
        for size in range(6):
            for parts in itertools.product(["<a", "</a>", ">", " ", "=", "/", "\t", "x"], repeat=size):
                page = "".join(parts)
                assert extract(page, cfg) == extract(_old_tags(page, "a"), bare), page
                seen += "<a=" in page or "<a\t" in page
        assert seen > 1000, "witness: the corpus holds tags followed by an equals sign or a tab"
        assert extract("<a=1>x</a> <a href=1>y</a>", cfg) == extract(_old_tags("<a=1>x</a> <a href=1>y</a>", "a"), bare)
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL37 -- a pattern that tries more ways from one start than the bound is refused
# ---------------------------------------------------------------------------
def test_sl37_a_pattern_that_tries_more_ways_from_one_start_than_the_bound_is_refused():
    lp, restore = _checker()
    try:
        for rounds in (13, 40, 100):
            assert lp.verdict(r"(?:x|(x)){%d}y" % rounds) == "exponential", rounds
        assert lp.verdict(r"(?:x|(x)){11}y") == "linear", "control: 2048 ways at most"
        assert lp.verdict(r"(?:a|ab){40}c") == "linear" and lp.verdict(r"(?:ab|cd){40}y") == "linear", \
            "control: branches that do not read one same word"
        few, more = re.compile(r"(?:x|(x)){8}y"), re.compile(r"(?:x|(x)){14}y")
        assert _cpu(more.search, "x" * 30) > 20 * _cpu(few.search, "x" * 30), \
            "witness: the time from one start doubles with each round, however short the text"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL38 -- one try of a lookaround is bounded by its ways times its steps
# ---------------------------------------------------------------------------
def test_sl38_one_try_of_a_lookaround_is_bounded_by_its_ways_times_its_steps():
    lp, restore = _checker()
    try:
        assert lp.verdict(r"(?!(?:x|(x)){8}(x){591}y)") == "unchecked", "256 ways of 600 steps"
        assert lp.verdict(r"(?=(?:a|aa){8})x") == "linear", "control: 256 ways of 16 steps"
        assert lp.verdict(r"(?=a{500})") == "linear", "control: one way of 500 steps"
        assert lp.verdict(r"(?=(?:x|(x)){4}(x){300})") == "unchecked", "16 ways of 304 steps, past the bound"
        assert lp.verdict(r"(?=(?:x|(x)){4}(x){200})") == "linear", "control: 16 ways of 204 steps"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL39 -- every empty path to one step is a way of its own
# ---------------------------------------------------------------------------
_SL39_OPTIONAL = r"(?:(?:(a?)|(b?)|(c?)|(d?)|(e?)|(f?)|(g?)|(h?))x){%d}y"


def test_sl39_every_empty_path_to_one_step_is_a_way_of_its_own():
    lp, restore = _checker()
    try:
        assert lp.verdict(_SL39_OPTIONAL % 12) == "exponential", "eight empty ways a round, twelve rounds"
        assert lp.verdict(_SL39_OPTIONAL % 3) == "linear", "control: three rounds, every way tried within the bound"
        assert lp.verdict(_SL39_OPTIONAL % 4) == "exponential", "one round past it"
        wide = "(?:(?:" + "()|" * 64 + "())x){%d}y"
        assert lp.verdict(wide % 2) == "exponential" and lp.verdict(wide % 1) == "linear", "65 ways a round, counted"
        assert lp.verdict(r"\b(?:\d(?: ?|-?|\.?)){13}\b") == "exponential" and lp.verdict(
            r"\b(?:\d(?: ?|-?|\.?)){5}\b") == "linear", "three empty ways a digit"
        few, more = re.compile(r"\A" + _SL39_OPTIONAL % 3), re.compile(r"\A" + _SL39_OPTIONAL % 5)
        assert _cpu(more.search, "x" * 20 + "zz") > 20 * _cpu(few.search, "x" * 20 + "zz"), \
            "witness: the engine tries each empty way, eight times more a round"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL40 -- the lookarounds one start may try take no more than the bound in all
# ---------------------------------------------------------------------------
def test_sl40_the_lookarounds_one_start_may_try_take_no_more_than_the_bound_in_all():
    lp, restore = _checker()
    try:
        look = r"(?!(?:x|(x)){8}x{7}y)"
        assert lp.verdict(look) == "linear", "control: one lookaround at the bound of one try"
        verdict, why = lp._judged(r"\A(?:" + look * 44 + r"(?:x|(x))){12}y", 0)
        assert verdict == "unchecked" and f"more than {lp.WAYS} steps in all" in why, why
        assert lp.verdict(r"(?:(?=a{10})\w){250}x") == "linear", "control: 250 tries of ten steps"
        verdict, why = lp._judged(r"\A(?:x|(x)){6}(?:" + look + r"x)*y", 0)
        assert verdict == "unchecked" and f"more than {lp.TRIES} steps" in why, ("tried again on each of the ways", why)
        assert lp.verdict(r"\A(?:x|(x)){2}(?:" + look + r"x)*y") == "linear", "control: tried on a few ways"
        verdict, why = lp._judged(r"(?:(?=a{20})\w){250}x", 0)
        assert verdict == "unchecked" and f"more than {lp.WAYS} steps in all" in why, ("250 tries of twenty steps", why)
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL41 -- a linear pattern a person may write for amounts is read within the check's bounds
# ---------------------------------------------------------------------------
def test_sl41_a_linear_pattern_for_amounts_is_read_within_the_check_s_bounds():
    lp, restore = _checker()
    try:
        amount = r"(?:\d{1,3}[ ,.]?){1,6}(?:\.\d{1,2})?\s?(?:EUR|USD|\$|" + chr(0x20AC) + ")"
        start = time.process_time()
        assert lp.verdict(amount) == "linear"
        assert time.process_time() - start < 2.0
        assert not _steep(re.compile(amount).search, lambda size: "1" * size), "as the engine reads it"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL42 -- an empty round of a loop goes on past the loop, a way of its own
# ---------------------------------------------------------------------------
_SL42_EMPTY_ROUNDS = [r"\A(?:(?:a?)*x)*y", r"^(?:(?:\s?)*\w)+$", r"\A(?:\d(?:[ ,]?)*)+\Z", r"(?:(?:a?)*x)*y"]


def test_sl42_an_empty_round_of_a_loop_goes_on_past_the_loop_a_way_of_its_own():
    lp, restore = _checker()
    try:
        for pattern in _SL42_EMPTY_ROUNDS:
            assert lp.verdict(pattern) == "exponential", pattern
        assert lp.verdict(r"\A(?:a?x)*y") == "linear", "control: a body that cannot match empty"
        engine = re.compile(_SL42_EMPTY_ROUNDS[0])
        assert _cpu(engine.search, "x" * 18) > 4 * _cpu(engine.search, "x" * 14), \
            "witness: the engine takes the empty round, then goes on: twice the ways for each character"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SL43 -- the empty ways to the end are tried where the end is refused, and again by every match in turn
# ---------------------------------------------------------------------------
def test_sl43_the_empty_ways_to_the_end_are_tried_where_the_end_is_refused_and_by_every_match_in_turn():
    lp, restore = _checker()
    try:
        for pattern in (r"(?:|){40}$", r"x(?:()|()){60}\b", r"(?:|){40}", r"(?:|){13}$"):
            assert lp.verdict(pattern) == "exponential", pattern
        assert lp.verdict(r"(?:|){11}$") == "linear", "control: 2048 empty ways"
        verdict, why = lp._judged(r"(?:|){12}(?![a-z]{0,60}!)", 0)
        assert verdict == "unchecked" and f"more than {lp.TRIES} steps" in why, ("each empty way tries the lookahead", why)
        every = re.compile(r"(?:|){16}")
        assert _cpu(every.findall, "x" * 20) > 20 * _cpu(every.search, "x" * 20), \
            "witness: after an empty match, every match in turn tries every empty way again"
        refused, few = re.compile(r"(?:|){16}$"), re.compile(r"(?:|){4}$")
        assert _cpu(refused.search, "x" * 20) > 20 * _cpu(few.search, "x" * 20), \
            "witness: where the end is refused, the engine tries every empty way to it"
    finally:
        restore()


# SL44 -- a state where the match may end still tries every empty way it has, to a step or to the end, before it ends

def test_sl44_a_state_where_the_match_may_end_still_tries_every_empty_way_it_has_before_it_ends():
    lp, restore = _checker()
    try:
        for pattern in (r"(?:(?:|){40}a)?", r"(?:(?:|){40}a)??", r"(?:a(?:(?:|){40}b)?)?", r"x(?:a(?:(?:|){40}b)?)?",
                        r"(?:a(?:(?:|){40}\b)?)?"):
            assert lp.verdict(pattern) == "exponential", pattern
        assert lp.verdict(r"(?:(?:|){11}a)?") == "linear", "control: 2049 empty ways"
        assert lp.verdict(r"(?:[a-z]+\s?)*") == "linear", "control: one way on, and an end at its first dead end"
        more, few = re.compile(r"(?:(?:|){16}a)?"), re.compile(r"(?:(?:|){4}a)?")
        assert _cpu(more.search, "b") > 20 * _cpu(few.search, "b"), \
            "witness: before it ends the match, the engine tries every empty way to the step"
        more, few = re.compile(r"(?:a(?:(?:|){16}b)?)?"), re.compile(r"(?:a(?:(?:|){4}b)?)?")
        assert _cpu(more.search, "ax") > 20 * _cpu(few.search, "ax"), \
            "witness: and so again where one such state reads into another"
        more, few = re.compile(r"(?:a(?:(?:|){18}\b)?)?"), re.compile(r"(?:a(?:(?:|){4}\b)?)?")
        assert _cpu(more.search, "ab") > 20 * _cpu(few.search, "ab"), \
            "witness: the empty ways to an end an assertion refuses are tried there too"
    finally:
        restore()


# SL45 -- the lookarounds tried on the ways of a state where the match may end are bounded too

def test_sl45_the_lookarounds_tried_on_the_ways_of_a_state_where_the_match_may_end_are_bounded():
    lp, restore = _checker()
    try:
        verdict, why = lp._judged(r"(?:(?:|){6}(?=[a-z]{0,60}!)a)?", 0)
        assert verdict == "unchecked" and f"more than {lp.TRIES} steps" in why, ("each empty way tries the lookahead", why)
        assert lp.verdict(r"(?:(?:|){4}(?=[a-z]{0,60}!)a)?") == "linear", "control: 17 ways"
        more, few = re.compile(r"(?:(?:|){10}(?=[a-z]{0,60}!)a)?"), re.compile(r"(?:(?:|){2}(?=[a-z]{0,60}!)a)?")
        assert _cpu(more.search, "a" * 60) > 20 * _cpu(few.search, "a" * 60), \
            "witness: every empty way tries the lookahead before the match ends"
    finally:
        restore()
