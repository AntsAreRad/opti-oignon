#!/usr/bin/env python3
"""Contracts for the guard that holds every regular expression of the package to linear time on a long run.

Python's engine backtracks: a pattern whose repeats can split one run of
characters in more than one way, or whose leading repeat is tried again from
every position of a run, takes quadratic time or worse on such a run, and a
page, a tool's output or a model's answer can be long and steered. The guard
reads every pattern of the package the source fixes -- literals, and what
constants, their loops and their joins build -- from the AST, pumps each
unbounded repeat behind a text that leads to it (a run of one character, that
text and the character repeated as a unit, one round of its body repeated),
times the reading at doubling lengths in the mode its callers use, and holds
each module's superlinear patterns to a ledger, by digest, that may only
shrink.

  * PG1 -- a quadratic pattern in a module the ledger does not name fails the
    guard by name.
  * PG2 -- the same pattern rewritten to read a run from its start only
    passes.
  * PG3 -- an owed module holding the patterns its entry names passes; one
    more fails.
  * PG4 -- an entry the census no longer finds superlinear is stale and fails.
  * PG5 -- a pattern compiled into a name whose every call is ``match`` is
    timed as one; passed elsewhere, or searched, it is timed as a search.
  * PG6 -- an opening marker repeated with no closing one is pumped as a
    unit: a pattern linear on every run of one character is still found.
  * PG7 -- a guard that cannot see its planted quadratic pattern fails as
    blind.
  * PG8 -- a pattern built from anything but constants is counted as
    dynamic, never measured.
  * PG9 -- the real tree satisfies its own guard, and the green line carries
    its denominators.
  * PG10 -- a pattern given to ``re`` by any name it is imported as, or to a
    function imported from it, is read.
  * PG11 -- a pattern built from constants is measured once per value it can
    take: concatenations, f-strings, joins, ``re.escape``, the items a loop or
    a comprehension takes from a constant, each at its own site.
  * PG12 -- a value that can change or come from outside is never taken for a
    constant: the site is counted dynamic, never measured and never lost.
  * PG13 -- the text leading to a repeat goes through the groups, branches,
    classes and repeats before it.
  * PG14 -- a class that accepts none of the candidate characters is pumped
    with one of its own.
  * PG15 -- possessive repeats, atomic groups and conditionals are pumped.
  * PG16 -- a compiled name matched from a position, imported under another
    name, or read from its module by another is timed as a search.
  * PG17 -- the ledger names patterns: one made linear cannot be traded for a
    new one, and one written twice is owed twice; a finding shows its
    pattern, cut at 80 characters.
  * PG18 -- only the package's own data directory, and bytecode caches, are
    pruned.
  * PG19 -- the doubling reads on past its short length while the time more
    than doubles, and stops there otherwise or at its ceiling.
  * PG20 -- each time is the least of two runs of the costliest method.
  * PG21 -- every function of ``re`` gives its pattern and its flags from their
    own positions or keywords.
  * PG22 -- the pumps of a pattern are its leads, characters and bodies.
  * PG23 -- a bytes pattern is measured, as Latin-1 text with ASCII classes.
  * PG24 -- the ledger never owes more than it owed when it was written.
  * PG25 -- a list, dict or set that anything could change out of the
    module's sight is never taken for a constant.
  * PG26 -- a verdict against the ledger counts when two of three readings
    agree: one noisy reading neither fails a linear pattern, nor makes an
    owed one stale, nor lets a superlinear one through.
  * PG27 -- a name bound where the evaluator cannot see its value -- by ``:=``
    inside a comprehension, by a ``match`` capture, by ``import *`` -- is
    never taken for a constant.
  * PG28 -- a name another module, ``setattr`` or ``globals()`` rebinds is
    never taken for a constant.
  * PG29 -- a list changed through an alias -- a mutating method taken as a
    value, an item stored through a loop variable -- is never taken for a
    constant.
  * PG30 -- ``re`` or one of its functions reached through an assignment is
    followed; read as a value the guard cannot follow, or called with
    ``**``, it is a dynamic site, never a silent one.
  * PG31 -- a module is read as Python reads it, its coding declaration
    honoured; one that cannot be parsed fails the guard by name.
  * PG32 -- a pattern that does not compile is counted dynamic, never
    dropped, and never stops the guard.
  * PG33 -- a back-reference stands for the group it names; a repeat no pump
    reaches is counted dynamic, never read as linear.
  * PG34 -- an exponential pattern ends its pump at its first burst and is
    found, never hangs the guard.
  * PG35 -- the pump characters the rest of the pattern refuses come first.
  * PG36 -- a reading near the bar, not owed, is read again before it passes.
  * PG37 -- every form of constant the evaluator reads gives the patterns it
    can take, and every form it cannot read is dynamic: a table of sites.
  * PG38 -- a change in place reaches a list through every way a name can
    share its objects, through two names in a row too; ``globals().update``
    opens a module; ``delattr`` on an imported module rebinds.
  * PG39 -- the pumps, the sample text, the first characters, the groups and
    the unbounded repeats of a table of patterns, one per kind of node the
    parser gives (classes, repeats lazy, possessive or bounded, groups,
    back-references, conditionals, lookarounds, atomic groups, anchors).
  * PG40 -- a pattern no pump finds superlinear is reported by its costliest
    pump, labelled by its kind (fake times, real loop).
  * PG41 -- the guard run from the command line reads the tree it is given and
    exits with the run's code, printing its lines.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolation import REPO, isolate  # noqa: E402

_GUARD = REPO / ".github" / "scripts" / "pattern_cost_guard.py"

BUDGET_S = {
    "test_pg1_a_quadratic_pattern_in_a_module_the_ledger_does_not_name_fails_by_name": 3.0,
    "test_pg2_the_same_pattern_reading_a_run_from_its_start_only_passes": 3.0,
    "test_pg3_an_owed_module_holding_as_many_as_its_entry_passes_and_one_more_fails": 3.0,
    "test_pg4_an_entry_above_its_module_s_count_is_stale_and_fails": 3.0,
    "test_pg5_a_name_called_only_with_match_is_timed_as_one_and_one_passed_elsewhere_as_a_search": 3.0,
    "test_pg6_an_opening_marker_repeated_with_no_closing_one_is_pumped_as_a_unit": 3.0,
    "test_pg7_a_guard_that_cannot_see_its_planted_quadratic_pattern_fails_as_blind": 3.0,
    "test_pg8_a_pattern_built_from_anything_but_literals_is_counted_as_dynamic": 3.0,
    "test_pg9_the_real_tree_satisfies_its_own_guard_with_its_denominators": 40.0,
    "test_pg10_a_pattern_given_to_re_by_another_name_or_through_an_imported_function_is_read": 3.0,
    "test_pg11_a_pattern_built_from_constants_is_measured_once_per_value_at_its_own_site": 3.0,
    "test_pg12_a_value_that_can_change_or_come_from_outside_is_dynamic_and_never_lost": 3.0,
    "test_pg13_the_text_leading_to_a_repeat_goes_through_groups_branches_classes_and_repeats": 3.0,
    "test_pg14_a_class_that_accepts_no_candidate_is_pumped_with_one_of_its_own": 3.0,
    "test_pg15_possessive_repeats_atomic_groups_and_conditionals_are_pumped": 3.0,
    "test_pg16_a_name_matched_from_a_position_or_imported_or_read_from_its_module_is_a_search": 5.0,
    "test_pg17_the_ledger_names_patterns_so_a_fixed_one_cannot_be_traded_for_a_new_one": 3.0,
    "test_pg18_only_the_package_s_own_data_directory_is_pruned": 3.0,
    "test_pg19_the_doubling_reads_past_its_short_length_only_while_the_time_more_than_doubles": 3.0,
    "test_pg20_each_time_is_the_least_of_two_runs_of_the_costliest_method": 3.0,
    "test_pg21_every_function_of_re_gives_its_pattern_and_flags_from_their_own_positions": 3.0,
    "test_pg22_the_pumps_of_a_pattern_are_its_leads_characters_and_bodies": 3.0,
    "test_pg23_a_bytes_pattern_is_measured_as_latin1_text_with_ascii_classes": 3.0,
    "test_pg24_the_ledger_never_owes_more_than_it_owed_when_it_was_written": 3.0,
    "test_pg25_a_list_dict_or_set_that_can_change_out_of_sight_is_never_taken_for_a_constant": 3.0,
    "test_pg26_a_verdict_against_the_ledger_counts_when_two_of_three_readings_agree": 3.0,
    "test_pg27_a_name_bound_where_the_evaluator_cannot_see_its_value_is_never_a_constant": 3.0,
    "test_pg28_a_name_another_module_setattr_or_globals_rebinds_is_never_a_constant": 3.0,
    "test_pg29_a_list_changed_through_an_alias_is_never_a_constant": 3.0,
    "test_pg30_re_reached_through_an_assignment_is_followed_and_a_loose_use_is_a_dynamic_site": 3.0,
    "test_pg31_a_module_is_read_as_python_reads_it_and_one_that_cannot_be_parsed_fails_by_name": 3.0,
    "test_pg32_a_pattern_that_does_not_compile_is_counted_dynamic_and_never_stops_the_guard": 3.0,
    "test_pg33_a_back_reference_stands_for_its_group_and_an_unpumped_repeat_is_dynamic": 3.0,
    "test_pg34_an_exponential_pattern_ends_its_pump_at_its_first_burst_and_is_found": 5.0,
    "test_pg35_the_pump_characters_the_rest_of_the_pattern_refuses_come_first": 3.0,
    "test_pg36_a_reading_near_the_bar_not_owed_is_read_again_before_it_passes": 3.0,
    "test_pg37_every_constant_form_gives_its_patterns_and_every_unread_form_is_dynamic": 3.0,
    "test_pg38_a_change_in_place_reaches_a_list_through_every_way_a_name_shares_its_objects": 3.0,
    "test_pg39_the_pump_helpers_give_their_exact_outputs_on_a_table_of_patterns": 3.0,
    "test_pg40_a_linear_pattern_is_reported_by_its_costliest_pump_labelled_by_its_kind": 3.0,
    "test_pg41_the_guard_run_from_the_command_line_reads_the_tree_it_is_given": 3.0,
}


def _load():
    loaded, restore = isolate(targets={"pattern_cost_guard": _GUARD})
    return loaded["pattern_cost_guard"], restore


def _tree(tmp_path, modules):
    """A package of ``modules`` (relative path -> source) under ``tmp_path``."""
    for rel, text in modules.items():
        path = tmp_path / "opti_oignon" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


_QUADRATIC_PATTERN = r"\s*\[\d+\]"
_QUADRATIC = 'import re\n_MARK = re.compile(r"\\s*\\[\\d+\\]")\n\n\ndef strip(text):\n    return _MARK.sub("", text)\n'
_LINEAR = ('import re\n_MARK = re.compile(r"(?:(?<!\\s)\\s+)?\\[\\d+\\]")\n\n\ndef strip(text):\n'
           '    return _MARK.sub("", text)\n')


# ---------------------------------------------------------------------------
# PG1 -- a quadratic pattern in a module the ledger does not name
# ---------------------------------------------------------------------------
def test_pg1_a_quadratic_pattern_in_a_module_the_ledger_does_not_name_fails_by_name(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = {}
        code, lines = guard.run(_tree(tmp_path, {"marks.py": _QUADRATIC}))
        assert code == 1, lines
        assert any("opti_oignon/marks.py: 1 superlinear pattern(s), 0 owed" in line for line in lines), lines
        assert any("line 2:" in line and "\\\\s*\\\\[" in line for line in lines), lines
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG2 -- the same pattern reading a run from its start only passes
# ---------------------------------------------------------------------------
def test_pg2_the_same_pattern_reading_a_run_from_its_start_only_passes(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = {}
        code, lines = guard.run(_tree(tmp_path, {"marks.py": _LINEAR}))
        assert code == 0, lines
        assert "1 measured" in lines[0] and "0 superlinear owed" in lines[0], lines
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG3 -- an owed module: as many as its entry pass, one more fails
# ---------------------------------------------------------------------------
def test_pg3_an_owed_module_holding_as_many_as_its_entry_passes_and_one_more_fails(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = {"opti_oignon/marks.py": (guard.digest(_QUADRATIC_PATTERN, 0),)}
        assert guard.run(_tree(tmp_path, {"marks.py": _QUADRATIC}))[0] == 0, "control: owed, held"
        more = _QUADRATIC + '_TAIL = re.compile(r"\\d+\\.\\s")\n\n\ndef listed(text):\n    return _TAIL.search(text)\n'
        code, lines = guard.run(_tree(tmp_path, {"marks.py": more}))
        assert code == 1 and any("2 superlinear pattern(s), 1 owed" in line for line in lines), lines
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG4 -- a stale entry fails
# ---------------------------------------------------------------------------
def test_pg4_an_entry_above_its_module_s_count_is_stale_and_fails(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = {"opti_oignon/marks.py": (guard.digest(_QUADRATIC_PATTERN, 0),)}
        code, lines = guard.run(_tree(tmp_path, {"marks.py": _LINEAR}))
        assert code == 1, lines
        assert any("the ledger owes 1, the census finds 0: lower the entry" in line for line in lines), lines
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG5 -- the mode a name is read in
# ---------------------------------------------------------------------------
def test_pg5_a_name_called_only_with_match_is_timed_as_one_and_one_passed_elsewhere_as_a_search(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = {}
        anchored = 'import re\n_PORT = re.compile(r".*:\\d+$")\n\n\ndef port(token):\n    return _PORT.match(token)\n'
        assert guard.run(_tree(tmp_path, {"ports.py": anchored}))[0] == 0, "matched only: one attempt, linear"
        passed = anchored + "\n\ndef table():\n    return [_PORT]\n"
        code, lines = guard.run(_tree(tmp_path, {"ports.py": passed}))
        assert code == 1 and any("ports.py: 1 superlinear" in line for line in lines), lines
        direct = 'import re\n\n\ndef port(token):\n    return re.search(r".*:\\d+$", token)\n'
        assert guard.run(_tree(tmp_path, {"ports.py": direct}))[0] == 1, "a search starts again at every position"
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG6 -- an opening marker repeated with no closing one
# ---------------------------------------------------------------------------
def test_pg6_an_opening_marker_repeated_with_no_closing_one_is_pumped_as_a_unit(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = {}
        think = ('import re\n_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)\n\n\ndef strip(text):\n'
                 '    return _THINK.sub("", text)\n')
        code, lines = guard.run(_tree(tmp_path, {"think.py": think}))
        assert code == 1, lines
        assert any("(<think>" in line and ")*" in line for line in lines), ("found by the unit pump", lines)
        result = _steep(guard, r"<think>.*?</think>", 16)
        assert result is not None and result[2].startswith("("), result
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG7 -- blind
# ---------------------------------------------------------------------------
def test_pg7_a_guard_that_cannot_see_its_planted_quadratic_pattern_fails_as_blind(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = {}
        tree = _tree(tmp_path, {"marks.py": _LINEAR})
        assert guard.run(tree)[0] == 0, "control: it sees"
        guard.GROWTH = 1000.0
        code, lines = guard.run(tree)
        assert code == 1 and "blind" in lines[0], lines
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG8 -- dynamic patterns
# ---------------------------------------------------------------------------
def test_pg8_a_pattern_built_from_anything_but_literals_is_counted_as_dynamic(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = {}
        built = ('import re\n\n\ndef find(text, word):\n    return re.search(r"\\s*" + word + r"\\s*=", text)\n')
        code, lines = guard.run(_tree(tmp_path, {"built.py": built}))
        assert code == 0, lines
        assert "1 site(s) read, 0 measured, 1 dynamic" in lines[0], lines
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG9 -- the real tree
# ---------------------------------------------------------------------------
def test_pg9_the_real_tree_satisfies_its_own_guard_with_its_denominators():
    guard, restore = _load()
    try:
        code, lines = guard.run(REPO)
        assert code == 0, lines
        line = lines[0]
        assert line.startswith("Pattern cost: OK --") and " measured, " in line and " dynamic, " in line, line
        measured = int(line.split(" read, ")[1].split(" measured")[0])
        assert measured > 300, ("the census reads the package", line)
        owed = sum(len(names) for names in guard.LEDGER.values())
        assert f"{owed} superlinear owed in {len(guard.LEDGER)} module(s)" in line, line
    finally:
        restore()


def _owing(guard, module, *patterns):
    """A ledger owing ``patterns`` (flags 0) in ``module`` of the planted package."""
    return {f"opti_oignon/{module}": tuple(guard.digest(pattern, 0) for pattern in patterns)}


def _steep(guard, pattern, flags=0, methods=("search",)):
    """A superlinear reading of a pattern when most of the guard's readings find it superlinear, as the guard reads
    a verdict against its ledger; else None."""
    verdict, reading = guard.steady(pattern, flags, methods)
    return reading if verdict else None


def _lines(lines):
    """The line numbers the findings of a failed run name, in order."""
    return [int(line.split("line ")[1].split(":")[0]) for line in lines if line.startswith("  line ")]


# ---------------------------------------------------------------------------
# PG10 -- re by another name
# ---------------------------------------------------------------------------
def test_pg10_a_pattern_given_to_re_by_another_name_or_through_an_imported_function_is_read(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = {}
        aliased = ('import re as rx\nfrom re import compile as build, search\n\n'
                   '_A = rx.compile(r"\\s*\\[\\d+\\]")\n_B = build(r"\\s*\\{\\d+\\}")\n\n\n'
                   'def strip(text):\n    return _A.sub("", _B.sub("", text)), search(r"\\s*<\\d+>", text)\n')
        code, lines = guard.run(_tree(tmp_path, {"aliased.py": aliased}))
        assert code == 1 and any("aliased.py: 3 superlinear pattern(s), 0 owed" in line for line in lines), lines
        assert _lines(lines) == [4, 5, 9], lines
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG11 -- patterns built from constants
# ---------------------------------------------------------------------------
_BUILT = '''import re

_OPEN = r"\\s*"
_MARKS = (r"\\[\\d+\\]", r"\\{\\d+\\}")
_NAMES = {"a": (r"<\\d+>",), "b": (r"\\(\\d+\\)",)}
_SEP = "|"


def strip(text):
    for mark in _MARKS:
        text = re.sub(_OPEN + mark, "", text)
    return text


def tags(text):
    return [re.search(_OPEN + name, text) for names in _NAMES.values() for name in names]


def joined(text):
    return re.search(f"{_OPEN}(?:{_SEP.join(_MARKS)})", text), re.search(re.escape(".") + _OPEN + "x", text)
'''


def test_pg11_a_pattern_built_from_constants_is_measured_once_per_value_at_its_own_site(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = {}
        tree = _tree(tmp_path, {"built.py": _BUILT})
        code, lines = guard.run(tree)
        assert code == 1 and _lines(lines) == [11, 11, 16, 16, 20], lines
        guard.LEDGER = _owing(guard, "built.py", r"\s*\[\d+\]", r"\s*\{\d+\}", r"\s*<\d+>", r"\s*\(\d+\)",
                              r"\s*(?:\[\d+\]|\{\d+\})")
        code, lines = guard.run(tree)
        assert code == 0, lines
        assert "4 site(s) read, 6 measured, 0 dynamic, 5 superlinear owed in 1 module(s)" in lines[0], lines
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG12 -- values that can change or come from outside
# ---------------------------------------------------------------------------
_CHANGING = '''import re

_GROWN = [r"\\s*\\[\\d+\\]"]
_GROWN.append(r"\\s*\\{\\d+\\}")
_EMPTY = ()
_SWAPPED = r"x"
_KEPT = [r"\\s*<\\d+>"]


def swap():
    global _SWAPPED
    _SWAPPED = r"\\s*<"


def given(text, pattern):
    return re.search(pattern, text)


def grown(text):
    return [re.search(p, text) for p in _GROWN]


def empty(text):
    return [re.search(p, text) for p in _EMPTY]


def swapped(text):
    return re.search(_SWAPPED, text)


def configured(text, config):
    return re.search(config.get("pattern", r"\\s*\\[\\d+\\]"), text)


def kept(text):
    return [re.search(p, text) for p in _KEPT]
'''


def test_pg12_a_value_that_can_change_or_come_from_outside_is_dynamic_and_never_lost(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = _owing(guard, "changing.py", r"\s*<\d+>")
        code, lines = guard.run(_tree(tmp_path, {"changing.py": _CHANGING}))
        assert code == 0, lines
        assert "6 site(s) read, 1 measured, 5 dynamic, 1 superlinear owed" in lines[0], ("control: the kept list", lines)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG13 -- the lead
# ---------------------------------------------------------------------------
def test_pg13_the_text_leading_to_a_repeat_goes_through_groups_branches_classes_and_repeats():
    guard, restore = _load()
    try:
        assert _steep(guard, r"(?:foo|baz).*bar"), "an opening branch repeated"
        assert _steep(guard, r"<a\s+href=.*?</a>"), "an opening holding a repeat, repeated"
        assert ("<a href=", " ", None) in guard.pumps(r"<a\s+href=.*?</a>", 0)
        assert guard.pumps(r"(?:foo|baz)bar", 0) == [], "control: no unbounded repeat, no pump"
        assert guard._lead(list(guard.sre_parse.parse(r"(?:ab|c)\d{2}(?=q)[xy]+z*")), 4, 0) == "ab00x"
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG14 -- a class's own characters
# ---------------------------------------------------------------------------
def test_pg14_a_class_that_accepts_no_candidate_is_pumped_with_one_of_its_own():
    guard, restore = _load()
    try:
        parse = guard.sre_parse.parse
        assert guard.first_chars(list(parse(r"[C-Y]")), 0) == ["C"]
        assert guard.first_chars(list(parse("`")), 0) == ["`"]
        assert guard.first_chars(list(parse(r"[^\s\S]")), 0) == [], "control: a class nothing can enter"
        assert _steep(guard, r"[A-Z]+[A-Z]+9"), "two repeats over capitals"
        assert _steep(guard, r"(`+)(.+?)(?<!`)\1(?!`)"), "a run of backticks"
        assert not _steep(guard, r"(?<![A-Z])[A-Z]+9"), "control: a run read from its start"
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG15 -- possessive repeats, atomic groups, conditionals
# ---------------------------------------------------------------------------
def test_pg15_possessive_repeats_atomic_groups_and_conditionals_are_pumped():
    guard, restore = _load()
    try:
        for pattern in (r"\s*+\[\d+\]", r"(?>\s*)\[\d+\]", r"(a)?(?(1)\s*|\s*)\[\d+\]"):
            assert guard.pumps(pattern, 0)[0][1] == " ", (pattern, guard.pumps(pattern, 0))
            assert _steep(guard, pattern), pattern
        assert guard.pumps(r"(a)?(?(1)b|c)\[x\]", 0) == [], "control: a conditional with no unbounded repeat"
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG16 -- names the mode cannot follow
# ---------------------------------------------------------------------------
_PORTS = 'import re\n_PORT = re.compile(r".*:\\d+$")\n\n\ndef port(token):\n    return _PORT.match(token{})\n'


def test_pg16_a_name_matched_from_a_position_or_imported_or_read_from_its_module_is_a_search(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = {}
        assert guard.run(_tree(tmp_path, {"ports.py": _PORTS.format("")}))[0] == 0, "control: matched from the start"
        assert guard.run(_tree(tmp_path, {"ports.py": _PORTS.format(", pos=0")}))[0] == 0, "from position 0"
        assert guard.run(_tree(tmp_path, {"ports.py": _PORTS.format(", 0")}))[0] == 0, "from position 0, by position"
        assert guard.run(_tree(tmp_path, {"ports.py": _PORTS.format(", endpos=5")}))[0] == 0, "an end is no start"
        named = _PORTS.format("") + "\n\ndef other(holder):\n    return holder._PORT\n"
        assert guard.run(_tree(tmp_path / "named", {"ports.py": named, "other.py": "import json\n\nX = json._PORT\n"}))[0] \
            == 1, "read from a module it imports, under its name"
        unrelated = "class Holder:\n    _PORT = None\n\n\ndef read(holder):\n    return holder._PORT\n"
        assert guard.run(_tree(tmp_path / "unrelated", {"ports.py": _PORTS.format(""), "other.py": unrelated}))[0] == 0, \
            "an attribute of an object that is no module follows nothing"
        both = _PORTS.format("") + "\n\ndef whole(token):\n    return _PORT.fullmatch(token)\n"
        assert guard.run(_tree(tmp_path, {"ports.py": both}))[0] == 0, "match and fullmatch, both anchored"
        code, lines = guard.run(_tree(tmp_path, {"ports.py": _PORTS.format(", 3")}))
        assert code == 1 and any("ports.py: 1 superlinear" in line for line in lines), ("from a position", lines)
        code, lines = guard.run(_tree(tmp_path, {"ports.py": _PORTS.format(""), "other.py":
                                                 "from opti_oignon.ports import _PORT as found\n"}))
        assert code == 1 and any("ports.py: 1 superlinear" in line for line in lines), ("imported, renamed", lines)
        code, lines = guard.run(_tree(tmp_path, {"other.py": "import opti_oignon.ports as ports\n\n\n"
                                                             "def find(text):\n    return ports._PORT.search(text)\n"}))
        assert code == 1 and any("ports.py: 1 superlinear" in line for line in lines), ("read from its module", lines)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG17 -- the ledger names patterns
# ---------------------------------------------------------------------------
def test_pg17_the_ledger_names_patterns_so_a_fixed_one_cannot_be_traded_for_a_new_one(tmp_path):
    guard, restore = _load()
    try:
        traded = 'import re\n_MARK = re.compile(r"\\s*\\{\\d+\\}")\n\n\ndef strip(text):\n    return _MARK.sub("", text)\n'
        guard.LEDGER = _owing(guard, "marks.py", _QUADRATIC_PATTERN)
        code, lines = guard.run(_tree(tmp_path, {"marks.py": traded}))
        assert code == 1, lines
        assert any("1 superlinear pattern(s), 1 owed, 1 not in the ledger" in line for line in lines), lines
        new = guard.digest(r"\s*\{\d+\}", 0)
        assert any(line.startswith(f"  line 2: {new} ") for line in lines), lines
        assert any(line.endswith(": " + ascii(r"\s*\{\d+\}")[1:-1]) for line in lines), ("shown whole", lines)
        assert any(f"the census finds 0: lower the entry, remove {guard.digest(_QUADRATIC_PATTERN, 0)}" in line
                   for line in lines), lines
        twice = _QUADRATIC + '_AGAIN = re.compile(r"\\s*\\[\\d+\\]")\n'
        code, lines = guard.run(_tree(tmp_path, {"marks.py": twice}))
        assert code == 1 and any("2 superlinear pattern(s), 1 owed, 1 not in the ledger" in line for line in lines)
        guard.LEDGER = _owing(guard, "marks.py", _QUADRATIC_PATTERN, _QUADRATIC_PATTERN)
        assert guard.run(_tree(tmp_path, {"marks.py": twice}))[0] == 0, "owed twice, held"
        long = r"\s*\[\d+\]" + "(?:" + "x" * 90 + ")?"
        guard.LEDGER = {}
        code, lines = guard.run(_tree(tmp_path / "long", {"marks.py": "import re\n_LONG = re.compile(" + repr(long) + ")\n"}))
        assert code == 1 and lines[-1].endswith(": " + ascii(long)[1:-1][:80]), lines
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG18 -- the data directory
# ---------------------------------------------------------------------------
def test_pg18_only_the_package_s_own_data_directory_is_pruned(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = {}
        code, lines = guard.run(_tree(tmp_path, {"data/kept.py": _QUADRATIC, "tools/data/read.py": _QUADRATIC,
                                                 "node_modules/vendored.py": _QUADRATIC,
                                                 "__pycache__/stale.py": _QUADRATIC}))
        assert code == 1 and any("opti_oignon/tools/data/read.py: 1 superlinear" in line for line in lines), lines
        assert any("opti_oignon/node_modules/vendored.py: 1 superlinear" in line for line in lines), lines
        assert not any("kept.py" in line or "stale.py" in line for line in lines), lines
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG19, PG20 -- the doubling and the clock (fake times, real loop)
# ---------------------------------------------------------------------------
def test_pg19_the_doubling_reads_past_its_short_length_only_while_the_time_more_than_doubles():
    guard, restore = _load()
    try:
        seen = []

        def clock(cost):
            def timed(compiled, text, methods):
                seen.append(len(text))
                return cost(len(text))
            return timed

        guard._timed = clock(lambda size: 0.004 * (size / 16384) ** 2)
        result = guard.growth(r"\s*x", 0)
        assert guard.superlinear(result) and result[0] > 3.9 and max(seen) == guard.LAST + 1, (result, max(seen))
        assert 0.004 * ((guard.SHORT + 1) / 16384) ** 2 < guard.FLOOR, "control: under the floor at the short length"
        seen.clear()
        guard._timed = clock(lambda size: 1e-7 * size)
        result = guard.growth(r"\s*x", 0)
        assert not guard.superlinear(result) and max(seen) == guard.SHORT + 1, (result, max(seen))
        assert len(seen) == 4 * 12, ("four characters, twelve lengths each, from 2 to 4,096", len(seen))
        seen.clear()
        guard._timed = clock(lambda size: 0.02 * (size / 64) ** 2)
        result = guard.growth(r"\s*x", 0)
        assert guard.superlinear(result) and max(seen) == 65 and result[1] >= guard.STOP, (result, seen)
        seen.clear()
        guard._timed = clock(lambda size: 0.0 if size < 10 else 1e-7 * size)
        result = guard.growth(r"\s*x", 0)
        assert not guard.superlinear(result) and max(seen) == guard.SHORT + 1, ("a reading of 0 divides nothing", result)
        seen.clear()
        guard._timed = clock(lambda size: 2.0 ** -6 if size >= 30 else 2.0 ** -10)
        result = guard.growth(r"\s*x", 0)
        assert result == (16.0, 2.0 ** -6, "", " ") and max(seen) == 33, ("a burst of sixteen exactly", result, seen)
        seen.clear()
        guard._timed = clock(lambda size: 0.001 * (size / 3) ** 3)
        result = guard.growth(r"\s*x", 0)
        assert guard.superlinear(result) and max(seen) == 9, ("three lengths are enough", result, seen)
        seen.clear()
        guard._timed = clock(lambda size: guard.STOP if size == 17 else guard.STOP * size / 32)
        result = guard.growth(r"\s*x", 0)
        assert result[1] == guard.STOP and max(seen) == 17, ("the stop reached exactly", result, seen)
        seen.clear()
        guard._timed = clock(lambda size: 2.0 ** -10 if size < 4 else 2.0 ** -6)
        result = guard.growth(r"\s*x", 0)
        assert result == (16.0, 2.0 ** -6, "", " ") and max(seen) == 5, ("a burst at the second reading", result, seen)
        seen.clear()
        guard._timed = clock(lambda size: guard.FLOOR / 32 if size < 4 else guard.FLOOR)
        result = guard.growth(r"\s*x", 0)
        assert result[1] == guard.FLOOR and max(seen) == 5, ("a burst at the floor exactly", result, seen)
        seen.clear()
        guard._timed = clock(lambda size: {3: 0.004, 5: 0.012}.get(size, 0.04))
        result = guard.growth(r"\s*x", 0)
        assert guard.superlinear(result) and max(seen) == 9, ("the stop waits for three lengths", result, seen)
        seen.clear()
        guard._timed = clock(lambda size: 1e-12 * size * size)
        result = guard.growth(r"\s*x", 0)
        assert not guard.superlinear(result) and result[0] > 3.9, ("steep but under the floor", result)
        assert guard.superlinear((guard.GROWTH, guard.FLOOR, "", " ")), "the bar itself is superlinear"
        assert not guard.superlinear((guard.GROWTH, guard.FLOOR * 0.99, "", " "))
        for growth_per_doubling in (2.75, guard.HINT):
            seen.clear()
            guard._timed = clock(lambda size: growth_per_doubling ** (size - 1).bit_length() * 2.0 ** -30)
            result = guard.growth(r"\s*x", 0)
            assert max(seen) == guard.LAST + 1 and not guard.superlinear(result), (growth_per_doubling, max(seen))
        seen.clear()
        late = {1025: 1e-6, 2049: 3e-6, 4097: 6.4e-6, 8193: 12.8e-6}
        guard._timed = clock(lambda size: late.get(size, size * 1e-9))
        result = guard.growth(r"\s*x", 0)
        assert max(seen) == 8193, ("the short stop reads two doublings, not the last one", max(seen))
    finally:
        restore()


def test_pg20_each_time_is_the_least_of_two_runs_of_the_costliest_method():
    guard, restore = _load()
    try:
        ticks, calls = [0.0, 5.0, 10.0, 13.0, 20.0, 21.0, 30.0, 32.0], []

        class Clock:
            @staticmethod
            def perf_counter():
                return ticks.pop(0)

        class Compiled:
            def match(self, text):
                calls.append("match")

            def search(self, text):
                calls.append("search")

        guard.time = Clock
        assert guard._timed(Compiled(), "x", ("match", "search")) == 3.0
        assert calls == ["match", "match", "search", "search"] and ticks == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG21 -- the positions of the pattern and the flags
# ---------------------------------------------------------------------------
_EVERY = '''import re


def every(t):
    re.compile(r"a", re.I)
    re.search(r"b", t, re.M)
    re.match(r"c", t, re.S)
    re.fullmatch(r"d", t, re.X)
    re.findall(r"e", t, re.A)
    re.finditer(r"f", t, re.U)
    re.split(r"g", t, 0, re.I | re.M)
    re.sub(r"h", "", t, 0, re.S)
    re.subn(r"i", "", t, 0, re.X)
    re.search(r"j", t, flags=re.I)
    re.compile(pattern=r"k", flags=re.M)
    re.search(r"l", t)
    re.sub(r"m", "", t, count=1)
'''


def test_pg21_every_function_of_re_gives_its_pattern_and_flags_from_their_own_positions(tmp_path):
    guard, restore = _load()
    try:
        import re
        rows = [(line, pattern, flags, methods) for _m, line, _c, pattern, flags, methods
                in guard.sites(_tree(tmp_path, {"every.py": _EVERY}))]
        search = ("search",)
        assert rows == [(5, "a", int(re.I), search), (6, "b", int(re.M), search), (7, "c", int(re.S), ("match",)),
                        (8, "d", int(re.X), ("fullmatch",)), (9, "e", int(re.A), search), (10, "f", int(re.U), search),
                        (11, "g", int(re.I | re.M), search), (12, "h", int(re.S), search), (13, "i", int(re.X), search),
                        (14, "j", int(re.I), search), (15, "k", int(re.M), search), (16, "l", 0, search),
                        (17, "m", 0, search)], rows
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG22 -- the pumps
# ---------------------------------------------------------------------------
def test_pg22_the_pumps_of_a_pattern_are_its_leads_characters_and_bodies():
    guard, restore = _load()
    try:
        import hashlib
        import re
        assert guard.pumps(r"ab\s*c", 0) == [("ab", " ", None), ("ab", "\t", None), ("ab", "\n", None),
                                             ("ab", chr(0x3000), None)]
        assert guard.pumps(r"x(?:<t>.)*y", re.S) == [("x", "<", "<t> ")]
        assert guard.pumps(r"[^\n]{2,}z", 0) == [("", " ", None), ("", "\t", None), ("", "a", None), ("", "0", None)]
        assert guard.pumps(r"a{1,5}b", 0) == [], "a bounded repeat is not pumped"
        assert guard.pumps(r"(?=\s*x)y", 0)[0] == ("", " ", None), "a lookahead's repeat is pumped"
        assert guard._texts("ab", " ", "<t>", 8) == {"run": "ab" + " " * 8 + "\x00", "unit": "ab ab \x00",
                                                     "body": "ab<t><t>\x00"}
        assert guard._texts("", "a", None, 4) == {"run": "aaaa\x00"}
        assert guard._texts("abc", " ", "<tt>", 2) == {"run": "abc  \x00", "unit": "abc \x00", "body": "abc<tt>\x00"}
        assert guard._sample(list(guard.sre_parse.parse(r"(?:ab|c)\d{2}[xy]+")), 0) == "ab00x"
        assert guard.digest(r"\s*", 0) == hashlib.sha256(b"0:\\s*").hexdigest()[:12], "the ledger's names hold"
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG23 -- bytes patterns
# ---------------------------------------------------------------------------
def test_pg23_a_bytes_pattern_is_measured_as_latin1_text_with_ascii_classes(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = {}
        planted = 'import re\n_MARK = re.compile(rb"\\s*\\[\\d+\\]")\n\n\ndef strip(data):\n    return _MARK.sub(b"", data)\n'
        code, lines = guard.run(_tree(tmp_path, {"raw.py": planted}))
        assert code == 1 and any("raw.py: 1 superlinear pattern(s), 0 owed" in line for line in lines), lines
        assert guard.digest(b"\\s*", 0) == guard.digest(r"\s*", 0)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG24 -- the ledger's ceiling
# ---------------------------------------------------------------------------
def test_pg24_the_ledger_never_owes_more_than_it_owed_when_it_was_written():
    guard, restore = _load()
    try:
        import re
        owed = sum(len(names) for names in guard.LEDGER.values())
        assert owed <= 74 and len(guard.LEDGER) <= 28, (owed, len(guard.LEDGER))
        for module, names in guard.LEDGER.items():
            assert (REPO / module).is_file(), module
            assert names and all(re.fullmatch(r"[0-9a-f]{12}", name) for name in names), (module, names)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG25 -- containers that can change out of sight
# ---------------------------------------------------------------------------
_ESCAPING = '''import re

_KEPT = [r"\\s*\\[\\d+\\]"]
_PASSED = [r"\\s*\\{\\d+\\}"]
_SHARED = [r"\\s*<\\d+>"]


def fill(patterns):
    patterns.append(r"x")


def setup():
    fill(_PASSED)


def kept(text):
    if "x" in _KEPT and len(_KEPT):
        pass
    return [re.search(p, text) for p in _KEPT + []]


def passed(text):
    return [re.search(p, text) for p in _PASSED]


def shared(text):
    return [re.search(p, text) for p in _SHARED]
'''


def test_pg25_a_list_dict_or_set_that_can_change_out_of_sight_is_never_taken_for_a_constant(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = _owing(guard, "escaping.py", r"\s*\[\d+\]")
        code, lines = guard.run(_tree(tmp_path, {"escaping.py": _ESCAPING,
                                                 "other.py": "from opti_oignon.escaping import _SHARED\n"}))
        assert code == 0, lines
        assert "3 site(s) read, 1 measured, 2 dynamic, 1 superlinear owed" in lines[0], ("control: the kept list", lines)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG26 -- a verdict against the ledger is read again (fake readings, real reconciling)
# ---------------------------------------------------------------------------
def test_pg26_a_verdict_against_the_ledger_counts_when_two_of_three_readings_agree(tmp_path):
    guard, restore = _load()
    try:
        steep, flat = (4.0, 0.01, "", " "), (2.0, 0.0005, "", " ")
        planted = {pattern: steep if quadratic else flat for pattern, _flags, quadratic in guard.PLANTED}
        owed, other = _QUADRATIC_PATTERN, r"(?:(?<!\s)\s+)?\[\d+\]"
        readings, asked = {}, []

        def reading(pattern, flags, methods=("search",)):
            if pattern in planted:
                return planted[pattern]
            asked.append(pattern)
            return readings[pattern].pop(0)

        guard.growth = reading
        guard.LEDGER = _owing(guard, "marks.py", owed)
        tree = _tree(tmp_path, {"marks.py": _QUADRATIC, "plain.py": _LINEAR})
        readings.update({owed: [flat, steep, steep], other: [steep, flat, flat]})
        code, lines = guard.run(tree)
        assert code == 0 and asked == [owed, other, owed, owed, other, other], (lines, asked)
        readings.update({owed: [flat, flat, steep], other: [steep, steep, flat]})
        asked.clear()
        code, lines = guard.run(tree)
        assert code == 1 and asked == [owed, other, owed, other], (lines, asked)
        assert any("plain.py: 1 superlinear pattern(s), 0 owed, 1 not in the ledger" in line for line in lines), lines
        assert any("the ledger owes 1, the census finds 0: lower the entry" in line for line in lines), lines
        readings.update({owed: [steep], other: [flat]})
        asked.clear()
        assert guard.run(tree)[0] == 0 and asked == [owed, other], ("a reading for the ledger is read once", asked)
        tail = r"\d+\.\s"
        both = _QUADRATIC + '_TAIL = re.compile(r"\\d+\\.\\s")\n\n\ndef listed(text):\n    return _TAIL.search(text)\n'
        readings.update({owed: [flat, steep, steep], tail: [steep, steep, steep], other: [flat]})
        code, lines = guard.run(_tree(tmp_path / "both", {"marks.py": both, "plain.py": _LINEAR}))
        assert code == 1 and any("marks.py: 2 superlinear pattern(s), 1 owed, 1 not in the ledger" in line
                                 for line in lines), lines
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG27 -- names bound where the evaluator cannot see
# ---------------------------------------------------------------------------
_UNSEEN = '''import re

P = r"a"
x = r"a"
q = r"a"
rest = r"a"


def starred(t):
    for first, *rest in ((r"b", r"c"),):
        return re.search(rest, t)


def walrus(t):
    return [re.search(q, t) for x in (r"b+", r"(c+)+") if (q := x)]


def captured(o, t):
    match o:
        case str() as P:
            pass
    return re.search(P, t)
'''
_STARRED = 'import re\nfrom opti_oignon.elsewhere import *\n\nS = r"a"\n\n\ndef find(t):\n    return re.search(S, t)\n'
_PLAIN = 'import re\n\nS = r"\\s*\\[\\d+\\]"\n\n\ndef find(t):\n    return re.search(S, t)\n'


def test_pg27_a_name_bound_where_the_evaluator_cannot_see_its_value_is_never_a_constant(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = _owing(guard, "plain.py", r"\s*\[\d+\]")
        code, lines = guard.run(_tree(tmp_path, {"unseen.py": _UNSEEN, "starred.py": _STARRED, "plain.py": _PLAIN}))
        assert code == 0, lines
        assert "5 site(s) read, 1 measured, 4 dynamic" in lines[0], ("control: the plain module", lines)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG28 -- names rebound elsewhere
# ---------------------------------------------------------------------------
_REBOUND = '''import re

P = r"x"
Q = r"x"
R = r"x"
K = r"\\s*\\[\\d+\\]"
globals()["R"] = r"(a+)+$"


def find(t):
    return re.search(P, t), re.search(Q, t), re.search(R, t), re.search(K, t)
'''
_REBINDER = ('import opti_oignon.rebound as m\n\nm.P = r"(a+)+$"\nsetattr(m, "Q", r"(a+)+$")\nholder = object()\n'
             'holder.K = 1\n')


def test_pg28_a_name_another_module_setattr_or_globals_rebinds_is_never_a_constant(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = _owing(guard, "rebound.py", r"\s*\[\d+\]")
        code, lines = guard.run(_tree(tmp_path, {"rebound.py": _REBOUND, "rebinder.py": _REBINDER}))
        assert code == 0 and "4 site(s) read, 1 measured, 3 dynamic" in lines[0], ("control: K", lines)
        opened = _REBOUND.replace('globals()["R"]', "globals()[name]").replace("import re", "import re\nname = 'R'")
        guard.LEDGER = {}
        code, lines = guard.run(_tree(tmp_path, {"rebound.py": opened, "rebinder.py": _REBINDER}))
        assert code == 0 and "4 site(s) read, 0 measured, 4 dynamic" in lines[0], ("any name may be rebound", lines)
        guard.LEDGER = _owing(guard, "rebound.py", r"\s*\[\d+\]")
        unknown = 'import opti_oignon.rebound as m\n\nname = "Q"\nsetattr(m, name, r"(a+)+$")\n'
        code, lines = guard.run(_tree(tmp_path / "unknown", {"rebound.py": _REBOUND, "rebinder.py": unknown}))
        assert code == 1 and any("the census finds 0" in line for line in lines), ("an unknown name opens all", lines)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG29 -- lists changed through an alias
# ---------------------------------------------------------------------------
_ALIASED = '''import re

L = [r"a+"]
add = L.append
M = [[r"a+", "b"]]
N = [r"\\s*\\[\\d+\\]"]


def grow():
    add(r"(b+)+$")
    for row in M:
        row[0] = r"(x+)+$"


def find(t):
    return [re.search(p, t) for p in L], [re.search(p, t) for p, q in M], [re.search(p, t) for p in N]
'''


def test_pg29_a_list_changed_through_an_alias_is_never_a_constant(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = _owing(guard, "aliased.py", r"\s*\[\d+\]")
        code, lines = guard.run(_tree(tmp_path, {"aliased.py": _ALIASED}))
        assert code == 0 and "3 site(s) read, 1 measured, 2 dynamic" in lines[0], ("control: N", lines)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG30 -- re reached through assignments, and loose uses
# ---------------------------------------------------------------------------
_REACHED = '''import re

R = re
s = re.search


def find(t, kw):
    R.search(r"\\s*\\[\\d+\\]", t)
    s(r"\\s*\\{\\d+\\}", t)
    re.search(**kw)
    keep(re)
    keep(s)
    return list(map(re.compile, kw))


R3 = R4
R4 = re
s3 = s4
s4 = re.search


def later(t):
    R3.search(r"\\s*<\\d+>", t)
    s3(r"\\s*\\(\\d+\\)", t)
'''


def test_pg30_re_reached_through_an_assignment_is_followed_and_a_loose_use_is_a_dynamic_site(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = {}
        code, lines = guard.run(_tree(tmp_path, {"reached.py": _REACHED}))
        assert code == 1 and _lines(lines) == [8, 9, 23, 24], lines
        guard.LEDGER = _owing(guard, "reached.py", r"\s*\[\d+\]", r"\s*\{\d+\}", r"\s*<\d+>", r"\s*\(\d+\)")
        code, lines = guard.run(_tree(tmp_path, {"reached.py": _REACHED}))
        assert code == 0 and "8 site(s) read, 4 measured, 4 dynamic" in lines[0], lines
        reverse = 'import re\n\nR5 = R6\nR6 = re\n\n\ndef later(t):\n    return R5.search(r"\\s*<\\d+>", t)\n'
        guard.LEDGER = {}
        code, lines = guard.run(_tree(tmp_path / "reverse", {"reverse.py": reverse}))
        assert code == 1 and _lines(lines) == [8], ("two rounds of the fixed point", lines)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG31 -- modules read as Python reads them
# ---------------------------------------------------------------------------
def test_pg31_a_module_is_read_as_python_reads_it_and_one_that_cannot_be_parsed_fails_by_name(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = {}
        tree = _tree(tmp_path, {"__init__.py": ""})
        (tree / "opti_oignon" / "latin.py").write_bytes(
            b"# -*- coding: latin-1 -*-\nimport re\nNAME = '\xe9'\n_MARK = re.compile(r'\\s*\\[\\d+\\]')\n")
        code, lines = guard.run(tree)
        assert code == 1 and any("latin.py: 1 superlinear pattern(s)" in line for line in lines), lines
        (tree / "opti_oignon" / "broken.py").write_text("def broken(:\n", encoding="utf-8")
        code, lines = guard.run(tree)
        assert code == 1 and "could not be parsed" in lines[0] and lines[1].strip() == "opti_oignon/broken.py", lines
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG32 -- patterns that do not compile
# ---------------------------------------------------------------------------
_UNCOMPILED = '''import re


def find(t, m):
    return re.search(b"a" if m else "a", t, re.L if m else 0), re.compile(r"(")
'''


def test_pg32_a_pattern_that_does_not_compile_is_counted_dynamic_and_never_stops_the_guard(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = {}
        code, lines = guard.run(_tree(tmp_path, {"uncompiled.py": _UNCOMPILED}))
        assert code == 0, lines
        assert "2 site(s) read, 3 measured, 2 dynamic" in lines[0], lines
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG33 -- back-references; repeats no pump reaches
# ---------------------------------------------------------------------------
def test_pg33_a_back_reference_stands_for_its_group_and_an_unpumped_repeat_is_dynamic(tmp_path):
    guard, restore = _load()
    try:
        assert guard.pumps(r"(a)(?:\1+)+b", 0), "a body led by a back-reference is pumped"
        assert guard.pumps(r"(.)\1*(?:\1x)*y", 0)
        assert _steep(guard, r"(a)(?:\1+)+b"), "and found"
        guard.LEDGER = {}
        blind = 'import re\n\n\ndef find(t):\n    return re.search(r"[^\\s\\S]+x", t)\n'
        code, lines = guard.run(_tree(tmp_path, {"blind.py": blind}))
        assert code == 0 and "1 site(s) read, 0 measured, 1 dynamic" in lines[0], lines
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG34 -- exponential patterns
# ---------------------------------------------------------------------------
def test_pg34_an_exponential_pattern_ends_its_pump_at_its_first_burst_and_is_found():
    guard, restore = _load()
    try:
        import time
        for pattern in (r"(a+)+b", r"(?:a+){2,9}b", r"(?:x|xx)+y", r"(?:\w+\s?)+:"):
            start = time.perf_counter()
            result = _steep(guard, pattern)
            assert result is not None and result[0] >= guard.BURST, (pattern, result)
            assert time.perf_counter() - start < 2.0, (pattern, "never hangs")
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG35 -- pump characters the continuation refuses
# ---------------------------------------------------------------------------
def test_pg35_the_pump_characters_the_rest_of_the_pattern_refuses_come_first():
    guard, restore = _load()
    try:
        chars = [ch for lead, ch, body in guard.pumps(r"(\w+)+[a0_\xe9]", 0) if lead == ""]
        assert chars and not set(chars[:1]) & set("a0_" + chr(0xE9)), chars
        assert _steep(guard, r"(\w+)+[a0_\xe9]"), "found with a character the end refuses"
        assert [ch for _l, ch, _b in guard.pumps(r"ab\s*c", 0)][:2] == [" ", "\t"], "control: nothing to refuse"
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG36 -- readings near the bar (fake readings, real reconciling)
# ---------------------------------------------------------------------------
def test_pg36_a_reading_near_the_bar_not_owed_is_read_again_before_it_passes(tmp_path):
    guard, restore = _load()
    try:
        steep, near, flat = (4.0, 0.01, "", " "), (2.8, 0.002, "", " "), (2.0, 0.0005, "", " ")
        planted = {pattern: steep if quadratic else flat for pattern, _flags, quadratic in guard.PLANTED}
        readings, asked = {}, []

        def reading(pattern, flags, methods=("search",)):
            if pattern in planted:
                return planted[pattern]
            asked.append(pattern)
            return readings[pattern].pop(0)

        guard.growth = reading
        guard.LEDGER = {}
        tree = _tree(tmp_path, {"plain.py": _LINEAR})
        other = r"(?:(?<!\s)\s+)?\[\d+\]"
        readings[other] = [near, steep, steep]
        code, lines = guard.run(tree)
        assert code == 1 and len(asked) == 3, (lines, asked)
        assert any("plain.py: 1 superlinear pattern(s), 0 owed, 1 not in the ledger" in line for line in lines), lines
        readings[other], asked[:] = [flat], []
        assert guard.run(tree)[0] == 0 and len(asked) == 1, ("control: a reading far from the bar is read once", asked)
        readings[other], asked[:] = [near, flat, flat], []
        assert guard.run(tree)[0] == 0 and len(asked) == 2, ("a reading near the bar the next denies passes", asked)
        readings[other], asked[:] = [(guard.HINT, 0.002, "", " "), flat, flat], []
        assert guard.run(tree)[0] == 0 and len(asked) == 2, ("a reading at the bar exactly is read again", asked)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG37 -- the forms the evaluator reads
# ---------------------------------------------------------------------------
_FORMS = 'import re\n\nA = r"\\s*"\nB = r"\\[\\d+\\]"\nT = (r"\\{\\d+\\}", r"<\\d+>")\nD = {"k": r"\\(\\d+\\)", "j": r"!\\d+!"}\nF = re.I | re.M\nC1 = B\nC2 = C1\nX1 = X2 = r"x"\nX1 = X2\nX2 = X1\nU, V = (r"a", r"b")\nS = tuple("abcdefghijklmnopqrstuvwxyz0123")\n\n\nclass Holder:\n    INNER = r"\\s*;\\d+;"\n\n    def find(self, t):\n        return re.search(INNER, t)\n\n\ndef find(t, m):\n    re.search("%s%s" % (A, B), t)\n    re.search(f"{A}{T[0]}", t)\n    re.search(f"{A!r}", t)\n    re.search(f"{A:>3}", t)\n    re.search(A + (T[1] if m else D["k"]), t)\n    for key, value in D.items():\n        re.search(A + value, t)\n    for key in sorted(D.keys()):\n        re.search(A + key, t)\n    re.search(A + "|".join(list(T)), t, F)\n    re.search(re.escape("+") + A + C2, t)\n    re.search(X1, t)\n    re.search(U, t)\n    for x in S:\n        for y in S:\n            re.search(x + y, t)\n    re.search(A + "".join(T[:1]), t)\n'
_MORE = 'import re\n\nT = (r"\\{\\d+\\}", r"<\\d+>")\nF = re.I | re.M\nif (W := r"\\s*\\[\\d+\\]"):\n    pass\nQ = r"q"\n\n\ndef Q():\n    pass\n\n\nW2 = r"\\s*<\\d+>"\nLAZY = lambda: (W2 := r"(a+)+$")  # noqa: E731\nG = r"\\s*"\nG += r"x"\nP2 = "a" if F else "ab"\nP3 = "bc" if F else "c"\n\n\ndef more(t, m):\n    re.search(W, t)\n    re.search(Q, t)\n    re.search(W2, t)\n    re.search(G, t)\n    for z in (r"a", r"b"):\n        pass\n    re.search(z, t)\n    for d in (r"\\s*\\[\\d+\\]", r"\\s*\\[\\d+\\]"):\n        re.search(d, t)\n    re.search(P2 + P3, t)\n    re.search(f"{T}", t)\n    re.search("x" if m else "x", t)\n'
_MORE_ROWS = [(23, '\\s*\\[\\d+\\]', 0), (24, None, None), (25, '\\s*<\\d+>', 0), (26, None, None), (29, 'a', 0), (29, 'b', 0), (31, '\\s*\\[\\d+\\]', 0), (32, 'abc', 0), (32, 'ac', 0), (32, 'abbc', 0), (33, None, None), (34, 'x', 0)]
_SCOPES = 'import re\n\n\ndef g(t, p=re.compile(r"\\s*\\[\\d+\\]")):\n    return p\n\n\ndef deco(x):\n    return x\n\n\n@deco(re.compile(r"\\s*\\{\\d+\\}"))\nclass Holder:\n    INNERX = r"\\s*<\\d+>"\n    PAT = re.compile(INNERX)\n\n\ndef loops(t):\n    for p in (r"a",):\n        p = r"b"\n        re.search(p, t)\n    for m in re.finditer(r"\\s*\\(\\d+\\)", t):\n        pass\n    found = [x for x in re.findall(r"\\s*!\\d+!", t)]\n    nested = [y for x in found for y in re.findall(r"\\s*;\\d+;", x)]\n    return found, nested\n'
_SCOPES_ROWS = [(4, '\\s*\\[\\d+\\]', 0), (12, '\\s*\\{\\d+\\}', 0), (15, '\\s*<\\d+>', 0), (21, 'a', 0), (21, 'b', 0), (22, '\\s*\\(\\d+\\)', 0), (24, '\\s*!\\d+!', 0), (25, '\\s*;\\d+;', 0)]
_BOUNDS = 'import re\n\nROWS = tuple("abcdefghijklmnop")\nCOLUMNS = tuple("abcdefghijklmnopqrstuvwxyz012345")\nWIDER = COLUMNS + ("6",)\n\n\ndef bounded(t):\n    for x in ROWS:\n        for y in COLUMNS:\n            re.search(x + y, t)\n    for x in ROWS:\n        for y in WIDER:\n            re.search(x + y, t)\n'
_FORMS_ROWS = [(21, None, None), (25, '\\s*\\[\\d+\\]', 0), (26, '\\s*\\{\\d+\\}', 0), (27, None, None), (28, None, None), (29, '\\s*<\\d+>', 0), (29, '\\s*\\(\\d+\\)', 0), (31, '\\s*\\(\\d+\\)', 0), (31, '\\s*!\\d+!', 0), (33, '\\s*k', 0), (33, '\\s*j', 0), (34, '\\s*\\{\\d+\\}|<\\d+>', 10), (35, '\\+\\s*\\[\\d+\\]', 0), (36, None, None), (37, None, None), (40, None, None), (41, '\\s*\\{\\d+\\}', 0)]


def test_pg37_every_constant_form_gives_its_patterns_and_every_unread_form_is_dynamic(tmp_path):
    guard, restore = _load()
    try:
        rows = [(line, pattern, flags) for _m, line, _c, pattern, flags, _methods
                in guard.sites(_tree(tmp_path, {"forms.py": _FORMS}))]
        assert rows == _FORMS_ROWS, rows
        rows = [(line, pattern, flags) for module, line, _c, pattern, flags, _methods
                in guard.sites(_tree(tmp_path / "more", {"more.py": _MORE})) if module.endswith("more.py")]
        assert rows == _MORE_ROWS, rows
        rows = [(line, pattern, flags) for _m, line, _c, pattern, flags, _methods
                in guard.sites(_tree(tmp_path / "scopes", {"scopes.py": _SCOPES}))]
        assert rows == _SCOPES_ROWS, ("defaults, decorators, class bodies, loop iterables, rebinding in a loop", rows)
        bounded = {}
        for _m, line, _c, pattern, _f, _methods in guard.sites(_tree(tmp_path / "bounds", {"bounds.py": _BOUNDS})):
            bounded.setdefault(line, []).append(pattern)
        assert [len(found) for found in bounded.values()] == [guard.MOST, 1], "as many values as MOST, then dynamic"
        assert None not in bounded[11] and bounded[14] == [None], bounded.keys()
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG38 -- changes in place through aliases; opened and removed names
# ---------------------------------------------------------------------------
_SHARED_OBJECTS = 'import re\n\nL1 = [[r"a"]]\nL2 = {"k": [r"a"]}\nL3 = [[r"a"]]\nL4 = [[r"a"]]\nL5 = [[r"a"]]\nL6 = [[r"a"]]\nL7 = [[r"a"]]\nL8 = [[r"a"]]\nL9 = [[r"a"]]\nL10 = [[r"a"]]\nL11 = list((r"a",))\nK = [r"\\s*\\[\\d+\\]"]\n\n\ndef change(m):\n    rows = [item for item in L1]\n    again = rows[0]\n    again.append(r"(x+)+$")\n    for key, value in L2.items():\n        value.append(r"(x+)+$")\n    built = L3 + []\n    built[0].append(r"(x+)+$")\n    for row in L4:\n        row[0] = r"(x+)+$"\n    with L5[0] as held:\n        held.append(r"(x+)+$")\n    either = L6[0] or []\n    either.append(r"(x+)+$")\n    picked = L7[0] if m else []\n    picked.append(r"(x+)+$")\n    pair = (L8[0], 1)\n    pair[0].append(r"(x+)+$")\n    box = {"k": L9[0]}\n    box["k"].append(r"(x+)+$")\n    alias = L10[0]\n    alias += [r"(x+)+$"]\n    L11.append(r"(x+)+$")\n\n\ndef find(t):\n    for p in L1[0]:\n        re.search(p, t)\n    for p in L2["k"]:\n        re.search(p, t)\n    for p in L3[0]:\n        re.search(p, t)\n    for p in L4[0]:\n        re.search(p, t)\n    for p in L5[0]:\n        re.search(p, t)\n    for p in L6[0]:\n        re.search(p, t)\n    for p in L7[0]:\n        re.search(p, t)\n    for p in L8[0]:\n        re.search(p, t)\n    for p in L9[0]:\n        re.search(p, t)\n    for p in L10[0]:\n        re.search(p, t)\n    for p in L11:\n        re.search(p, t)\n    for p in K:\n        re.search(p, t)\n'


def test_pg38_a_change_in_place_reaches_a_list_through_every_way_a_name_shares_its_objects(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = _owing(guard, "shared.py", r"\s*\[\d+\]")
        code, lines = guard.run(_tree(tmp_path, {"shared.py": _SHARED_OBJECTS}))
        assert code == 0 and "12 site(s) read, 1 measured, 11 dynamic" in lines[0], ("control: K", lines)
        guard.LEDGER = {}
        code, lines = guard.run(_tree(tmp_path / "apart", {"opened.py": 'import re\n\nP = r"x"\nglobals().update({})\n\n\ndef find(t):\n    return re.search(P, t)\n', "target.py": 'import re\n\nQ = r"x"\n\n\ndef find(t):\n    return re.search(Q, t)\n',
                                                 "remover.py": 'import opti_oignon.target as m\n\ndelattr(m, "Q")\n'}))
        assert code == 0 and "2 site(s) read, 0 measured, 2 dynamic" in lines[0], lines
        itself = 'import re\n\nx = [r"a"]\nx.append(r"b")\nx = x\n\n\ndef find(t):\n    return [re.search(p, t) for p in x]\n'
        code, lines = guard.run(_tree(tmp_path / "itself", {"itself.py": itself}))
        assert code == 0 and "1 site(s) read, 0 measured, 1 dynamic" in lines[0], ("a name bound to itself", lines)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG39 -- the pump helpers on a table of patterns
# ---------------------------------------------------------------------------
# (pattern, flags, pumps, sample, first characters (six at most), groups, unbounded)
_PUMP_TABLE = [
    ('x[^\\n]*?y', 0, [('x', ' ', None), ('x', '\t', None), ('x', 'a', None), ('x', '0', None)], 'x y', ['x'], [], True),
    ('[a-c\\d]+z', 0, [('', 'a', None), ('', '0', None), ('', 'b', None), ('', '9', None)], 'az', ['a', '0', 'b', '9'], [], True),
    ('[^a]+a', 0, [('', ' ', None), ('', '\t', None), ('', '\n', None), ('', '0', None)], ' a', [' ', '\t', '\n', '0', '_', '.'], [], True),
    ('\\W+\\w', 0, [('', ' ', None), ('', '\t', None), ('', '\n', None), ('', '.', None)], ' a', [' ', '\t', '\n', '.', '-', ':'], [], True),
    ('(?i)K+q', 0, [('', 'K', None)], 'Kq', ['K'], [], True),
    ('a.+?b', 16, [('a', ' ', None), ('a', '\t', None), ('a', '\n', None), ('a', 'a', None)], 'a b', ['a'], [], True),
    ('(?:ab)+c', 0, [('', 'a', 'ab')], 'abc', ['a'], [], True),
    ('(?P<n>a+)(?P=n)', 0, [('', 'a', None)], 'aa', ['a'], [1], True),
    ('(a)?(?(1)b+|c+)d', 0, [('a', 'b', None), ('a', 'c', None)], 'abd', ['a'], [1], True),
    ('(a)?(?(1)b+)d', 0, [('a', 'b', None)], 'abd', ['a'], [1], True),
    ('(?=x+)y', 0, [('', 'x', None)], 'y', ['y'], [], True),
    ('(?<!z)q+r', 0, [('', 'q', None)], 'qr', ['q'], [], True),
    ('(?>ab+)c', 0, [('a', 'b', None)], 'abc', ['a'], [], True),
    ('a*+b', 0, [('', 'a', None)], 'ab', ['a'], [], True),
    ('^\\s+$', 8, [('', ' ', None), ('', '\t', None), ('', '\n', None), ('', '\u3000', None)], ' ', [' ', '\t', '\n', '\u3000', '\u2028'], [], True),
    ('(?:ab|a)+c', 0, [('', 'a', 'ab')], 'abc', ['a'], [], True),
    ('((a)(b))+\\2\\3', 0, [('', 'a', 'ab')], 'abab', ['a'], [1, 2, 3], True),
    ('[\u0400-\u04ff]+x', 0, [('', '\u0400', None)], '\u0400x', ['\u0400'], [], True),
    ('\\d{2,}x', 0, [('', '0', None), ('', '9', None)], '00x', ['0', '9'], [], True),
    ('(?:x|y{3})+z', 0, [('', 'x', None), ('', 'y', None)], 'xz', ['x', 'y'], [], True),
    ('a{2,5}b', 0, [], 'aab', ['a'], [], False),
    ('(?:ab){3}', 0, [], 'ababab', ['a'], [], False),
    ('(?:a+){2}b', 0, [('', 'a', None)], 'aab', ['a'], [], True),
    ('x', 0, [], 'x', ['x'], [], False),
    ('[^\\s\\S]+x', 0, [], None, [], [], True),
    ('(?:[^\\s\\S]a)+b', 0, [], None, [], [], True),
    ('(a|[^\\s\\S])+c', 0, [('', 'a', None)], 'ac', ['a'], [1], True),
    ('(?:[^\\s\\S]|[^\\s\\S])x', 0, [], None, [], [], False),
    ('([^\\s\\S])\\1+', 0, [], None, [], [1], True),
    ('(?>[^\\s\\S])x+', 0, [('', 'x', None)], None, [], [], True),
    ('x(?:a|b+)', 0, [('x', 'b', None)], 'xa', ['x'], [], True),
    ('(\\w+)+[a0_\\xe9]', 0, [('', 'b', None), ('', 'x', None), ('', 'Z', None), ('', '9', None)], 'aa', ['a', '0', '_', '\xe9', 'b', 'x'], [1], True),
    ('[ab]+a', 0, [('', 'b', None), ('', 'a', None)], 'aa', ['a', 'b'], [], True),
    ('(?=c+)c+', 0, [('', 'c', None)], 'c', ['c'], [], True),
    ('(a)?(?(1)b+|c)', 0, [('a', 'b', None)], 'ab', ['a'], [1], True),
    ('(?>(a)b+)\\1', 0, [('a', 'b', None)], 'aba', ['a'], [1], True),
    ('(?=(a))\\1+', 0, [('', 'a', None)], 'a', ['a'], [1], True),
    ('(?:(a)x|b)\\1+', 0, [('ax', 'a', None)], 'axa', ['a', 'b'], [1], True),
    ('(a)?(?(1)(b)x|c)\\2+', 0, [('abx', 'b', None)], 'abxb', ['a'], [1, 2], True),
    ('(?i:a)b+', 0, [('a', 'b', None)], 'ab', ['a'], [], True),
    ('(a)?(?(1)[^\\s\\S])d', 0, [], 'ad', ['a'], [1], False),
    ('(a)?(?(1)[^\\s\\S]x|[^\\s\\S]y)d', 0, [], None, ['a'], [1], False),
    ('(?:[^\\s\\S]a|[^\\s\\S]b)x', 0, [], None, [], [], False),
    ('(?:([^\\s\\S])x|a)\\1', 0, [], None, ['a'], [1], False),
    ('[\u0416\u042f]+x', 0, [('', '\u0416', None)], '\u0416x', ['\u0416'], [], True),
    ('(a)?[ab]+(?(1)a|q)', 0, [('a', 'b', None), ('a', 'a', None)], 'aaa', ['a'], [1], True),
    ('^$', 0, [], '', [], [], False),
    ('(a)?(?(1)b)c+', 0, [('ab', 'c', None)], 'abc', ['a'], [1], True),
    ('(?:a+b|ac)x', 0, [('', 'a', None)], 'abx', ['a'], [], True),
]


def test_pg39_the_pump_helpers_give_their_exact_outputs_on_a_table_of_patterns():
    guard, restore = _load()
    try:
        assert len(_PUMP_TABLE) >= 20, "control: the table holds every kind of node"
        for pattern, flags, pumps, sample, first, groups, unbounded in _PUMP_TABLE:
            tree = list(guard.sre_parse.parse(pattern, flags))
            found = guard._groups(tree)
            assert guard.pumps(pattern, flags) == pumps, pattern
            assert guard._sample(tree, flags, found) == sample, pattern
            assert guard.first_chars(tree, flags, found)[:6] == first, pattern
            assert sorted(found) == groups, pattern
            assert guard.unbounded(pattern, flags) is unbounded, pattern
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG40 -- the costliest pump of a linear pattern (fake times, real loop)
# ---------------------------------------------------------------------------
def test_pg40_a_linear_pattern_is_reported_by_its_costliest_pump_labelled_by_its_kind():
    guard, restore = _load()
    try:
        def timed(compiled, text, methods):
            return 1e-7 * len(text) * (2 if "\t" in text else 1) * (3 if text.count("ab") > 1 else 1)

        guard._timed = timed
        result = guard.growth(r"ab\s*c", 0)
        assert not guard.superlinear(result), result
        assert result[2:] == ("(ab\t)*", "\t") and abs(result[0] - 2.0) < 1e-9, result
        assert abs(result[1] - 1e-7 * 4096 * 2 * 3) < 1e-12, ("the unit pump at 4,096", result)

        def mixed(compiled, text, methods):
            return 9.7e-6 * len(text) if "\t" in text else 2.5e-9 * len(text) ** 2

        guard._timed = mixed
        result = guard.growth(r"ab\s*c", 0)
        assert guard.superlinear(result) and result[2:] == ("ab", " "), ("the first superlinear pump, at once", result)
        guard._timed = lambda compiled, text, methods: 1e-7 * len(text)
        result = guard.growth(r"ab\s*c", 0)
        assert result[2:] == ("ab", " "), ("equal costs keep the first pump", result)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG41 -- the command line
# ---------------------------------------------------------------------------
def test_pg41_the_guard_run_from_the_command_line_reads_the_tree_it_is_given(tmp_path, capsys):
    guard, restore = _load()
    try:
        guard.LEDGER = {}
        assert guard.main(["pattern_cost_guard.py", str(_tree(tmp_path / "steep", {"marks.py": _QUADRATIC}))]) == 1
        out = capsys.readouterr().out.splitlines()
        assert out[0] == "Pattern cost: FAILED --" and "marks.py: 1 superlinear" in out[1], out
        assert guard.main(["pattern_cost_guard.py", str(_tree(tmp_path / "flat", {"marks.py": _LINEAR}))]) == 0
        out = capsys.readouterr().out.splitlines()
        assert len(out) == 1 and out[0].startswith("Pattern cost: OK -- 1 site(s) read, 1 measured"), out
        seen = []
        guard.run = lambda root: (seen.append(root), (0, ["read"]))[1]
        assert guard.main(["pattern_cost_guard.py"]) == 0 and seen == [REPO], ("no tree given: the repository", seen)
        import subprocess
        steep = _tree(tmp_path / "script", {"marks.py": _QUADRATIC})
        done = subprocess.run([sys.executable, str(_GUARD), str(steep)], capture_output=True, text=True, timeout=120)
        assert done.returncode == 1 and done.stdout.startswith("Pattern cost: FAILED --"), (done.returncode, done.stdout)
    finally:
        restore()
