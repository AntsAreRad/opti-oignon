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
  * PG42 -- ``chr`` of a constant integer is read: a pattern built with it is
    measured and held to the ledger; ``chr`` of anything else is dynamic.
  * PG43 -- a builtin whose name a scope binds, or another module rebinds, is
    never read as the builtin: the site is dynamic.
  * PG44 -- the ledger never owes more than it owes now that thirty-two of
    its patterns are linear: 42 patterns in 25 modules. It replaces PG24.
  * PG45 -- a package that names ``__builtins__``, ``exec`` or ``eval``, a
    frame's builtins, imports from ``builtins`` or uses it as an object, or
    writes through a module it imports by name at run time, has no builtin read
    as itself; a module only read so opens nothing; ``exec`` or ``eval`` opens
    its own module's names; a
    ``type`` alias or a type parameter binds its name (Python 3.12 and later:
    the syntax does not parse before).
  * PG46 -- the ledger is empty: every pattern the guard measures reads a long
    run in linear time, and any superlinear one fails the guard by name. It
    replaces PG44.
  * PG47 -- a branch that takes a run at once only where the real match
    fails, and refuses a character the real match takes, is found reading
    every match (``sub``, ``findall``, ``finditer``, ``split``); one search
    reads it in linear time.
  * PG48 -- a class of characters no character of the fixed list shows is
    pumped by one of its own.
  * PG49 -- the classes of a pattern are the partition of every code point
    by the sets its items accept under their own flags, with the words of a
    word boundary and the newline of a line anchor, read from the engine;
    one character shows each.
  * PG50 -- the doubling reads past its short length only while the time
    more than doubles, once for each class the body takes. It replaces PG19.
  * PG51 -- every function of ``re`` gives its pattern and its flags from
    their own positions or keywords, and the method it is read with. It
    replaces PG21.
  * PG52 -- the pumps of a pattern are its leads, one character for each
    class its bodies take, and its bodies. It replaces PG22.
  * PG53 -- the classes the rest of the pattern refuses come first. It
    replaces PG35.
  * PG54 -- every form of constant the evaluator reads gives the patterns it
    can take, ``sorted`` read sorted, and every form it cannot read is
    dynamic. It replaces PG37.
  * PG55 -- the pump helpers on the table of PG39, with the pumps of one
    character for each class. It replaces PG39.
  * PG56 -- a pattern no pump finds superlinear is reported by its costliest
    pump, labelled by its kind. It replaces PG40.
  * PG57 -- a compiled name is read with the costliest method its module
    calls it with: ``findall`` for any but a search or a match.
  * PG58 -- a group's type flag (ASCII, LOCALE, UNICODE) takes the place
    of the pattern's, as the engine combines them.
  * PG59 -- a caret under multiline mode reads the newline as a class of
    its own.
  * PG60 -- a match call that unpacks its arguments is timed as a search.
  * PG61 -- a lead before the repeat is written to pass an anchor or a
    lookaround in it.
  * PG62 -- a pattern stored on an object and read in another module is
    timed the costliest way.
  * PG63 -- a lead takes the preferred class only where its item does,
    and before a word boundary too.
  * PG64 -- a lead writes the branch that holds the preferred class, and a
    class none of the candidates shows.
  * PG65 -- a reading of the tree keeps none of the modules it parsed.
  * PG66 -- each time is the least of two runs of the costliest method, on
    the thread's own clock: the wall clock is never read. It replaces PG20.
  * PG67 -- a reading counts the time its thread runs, never the time it
    waits: a wait alone reads as nothing, and work as itself.
  * PG68 -- an exponential pattern ends its pump at its first burst and is
    found within two seconds of its thread's time. It replaces PG34.
  * PG69 -- where the longest reading takes 1 ms or more and the growth
    reads under the bar, the shorter of the two readings it compares is read
    twice more, keeping its least time: a slow spell in it never passes a
    quadratic pattern as linear, at the stop as at the ceiling; a growth
    read over the bar, or under the floor, is not read again.
"""
import gc
import sys
import weakref
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
    "test_pg42_chr_of_a_constant_integer_is_read_and_chr_of_anything_else_is_dynamic": 3.0,
    "test_pg43_a_builtin_a_scope_binds_or_another_module_rebinds_is_never_read_as_the_builtin": 3.0,
    "test_pg44_the_ledger_never_owes_more_than_it_owes_now_that_thirty_two_patterns_are_linear": 3.0,
    "test_pg45_a_package_that_can_rebind_the_builtins_reads_none_as_itself": 3.0,
    "test_pg46_the_ledger_is_empty_so_any_superlinear_pattern_fails_by_name": 3.0,
    "test_pg47_a_branch_that_refuses_a_character_the_real_match_takes_is_found_reading_every_match": 6.0,
    "test_pg48_a_class_no_character_of_the_fixed_list_shows_is_pumped_by_one_of_its_own": 3.0,
    "test_pg49_the_classes_of_a_pattern_are_read_from_the_engine_over_every_code_point": 3.0,
    "test_pg50_the_doubling_reads_past_its_short_length_only_while_the_time_more_than_doubles": 3.0,
    "test_pg51_every_function_of_re_gives_its_pattern_its_flags_and_the_method_it_is_read_with": 3.0,
    "test_pg52_the_pumps_of_a_pattern_are_its_leads_the_classes_its_bodies_take_and_its_bodies": 3.0,
    "test_pg53_the_pump_characters_the_rest_of_the_pattern_refuses_come_first": 3.0,
    "test_pg54_every_constant_form_gives_its_patterns_sorted_read_sorted_and_every_unread_form_is_dynamic": 3.0,
    "test_pg55_the_pump_helpers_give_their_exact_outputs_on_a_table_of_patterns": 3.0,
    "test_pg56_a_linear_pattern_is_reported_by_its_costliest_pump_labelled_by_its_kind": 3.0,
    "test_pg57_a_compiled_name_is_read_with_the_costliest_method_its_module_calls_it_with": 3.0,
    "test_pg58_a_group_s_type_flag_takes_the_place_of_the_pattern_s_as_the_engine_combines_them": 10.0,
    "test_pg59_a_caret_under_multiline_mode_reads_the_newline_as_a_class_of_its_own": 10.0,
    "test_pg60_a_match_call_that_unpacks_its_arguments_is_timed_as_a_search": 3.0,
    "test_pg61_a_lead_before_the_repeat_is_written_to_pass_an_anchor_or_a_lookaround_in_it": 15.0,
    "test_pg62_a_pattern_stored_on_an_object_and_read_in_another_module_is_timed_the_costliest_way": 6.0,
    "test_pg63_a_lead_takes_the_preferred_class_only_where_its_item_does_and_before_a_word_boundary": 10.0,
    "test_pg64_a_lead_writes_the_branch_that_holds_the_preferred_class_and_a_class_no_candidate_shows": 10.0,
    "test_pg65_a_reading_of_the_tree_keeps_none_of_the_modules_it_parsed": 3.0,
    "test_pg66_each_time_is_the_least_of_two_runs_of_the_costliest_method_on_the_threads_own_clock": 3.0,
    "test_pg67_a_reading_counts_the_time_its_thread_runs_never_the_time_it_waits": 3.0,
    "test_pg68_an_exponential_pattern_ends_its_pump_at_its_first_burst_and_is_found_within_its_threads_time": 5.0,
    "test_pg69_a_growth_read_under_the_bar_over_the_floor_reads_its_shorter_length_again": 3.0,
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


# ---------------------------------------------------------------------------
# PG42 -- chr of a constant integer
# ---------------------------------------------------------------------------
_CHARS = ('import re\n\nBLANK = " " + chr(0xA0) + chr(0x202F)\nTOO_FAR = chr(0x110000)\n\n\ndef find(t, n):\n'
          '    re.search("[" + BLANK + r"]+\\[\\d+\\]", t)\n    re.search(chr(97) + "+", t)\n    re.search(chr(n), t)\n'
          '    re.search(chr("a"), t)\n    re.search(chr(True), t)\n    re.search(TOO_FAR, t)\n'
          '    re.search(chr(0) + "+", t)\n    re.search(chr(0x10FFFF) + "+", t)\n    re.search(str(97) + "+", t)\n')


def test_pg42_chr_of_a_constant_integer_is_read_and_chr_of_anything_else_is_dynamic(tmp_path):
    guard, restore = _load()
    try:
        tree = _tree(tmp_path, {"chars.py": _CHARS})
        rows = [(line, pattern) for _m, line, _c, pattern, _f, _methods in guard.sites(tree)]
        blank = "[ " + chr(0xA0) + chr(0x202F) + "]+\\[\\d+\\]"
        assert rows == [(8, blank), (9, "a+"), (10, None), (11, None), (12, None), (13, None), (14, chr(0) + "+"),
                        (15, chr(0x10FFFF) + "+"), (16, None)], ("the bounds of chr; str is not chr", rows)
        guard.LEDGER = {}
        code, lines = guard.run(tree)
        assert code == 1 and _lines(lines) == [8], ("built by chr, measured and held to the ledger", lines)
        guard.LEDGER = _owing(guard, "chars.py", blank)
        assert guard.run(tree)[0] == 0, "owed, it passes"
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG43 -- builtins a scope binds or another module rebinds
# ---------------------------------------------------------------------------
_SHADOWS = ('import re\nT = ("a", "b")\n\n\ndef chr(code):\n    return "b"\n\n\ndef find(t, sorted=None):\n'
            '    re.search(chr(97), t)\n    re.search("|".join(sorted(T)), t)\n\n\ndef other(t):\n    tuple = list\n'
            '    return re.search("".join(tuple(T)), t)\n\n\ndef fine(t):\n    return re.search("".join(list(T)), t)\n')
_USER = 'import re\n\n\ndef find(t):\n    return re.search(chr(97) + "+", t)\n'
_BUILTINS_REBOUND = 'import builtins\n\nbuiltins.chr = str\n'


def test_pg43_a_builtin_a_scope_binds_or_another_module_rebinds_is_never_read_as_the_builtin(tmp_path):
    guard, restore = _load()
    try:
        rows = [(line, pattern) for _m, line, _c, pattern, _f, _methods
                in guard.sites(_tree(tmp_path / "shadows", {"shadows.py": _SHADOWS}))]
        assert rows == [(10, None), (11, None), (16, None), (20, "ab")], rows
        alone = [pattern for _m, _l, _c, pattern, _f, _methods in guard.sites(_tree(tmp_path / "alone", {"user.py": _USER}))]
        assert alone == ["a+"], ("control: the builtin is read where nothing rebinds it", alone)
        rebound = [pattern for module, _l, _c, pattern, _f, _methods
                   in guard.sites(_tree(tmp_path / "rebound", {"user.py": _USER, "rebinder.py": _BUILTINS_REBOUND}))
                   if module.endswith("user.py")]
        assert rebound == [None], ("rebound by another module", rebound)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG44 -- the ledger's ceiling, lowered (replaces PG24)
# ---------------------------------------------------------------------------
def test_pg44_the_ledger_never_owes_more_than_it_owes_now_that_thirty_two_patterns_are_linear():
    guard, restore = _load()
    try:
        import re
        owed = sum(len(names) for names in guard.LEDGER.values())
        assert owed <= 42 and len(guard.LEDGER) <= 25, (owed, len(guard.LEDGER))
        for module, names in guard.LEDGER.items():
            assert (REPO / module).is_file(), module
            assert names and all(re.fullmatch(r"[0-9a-f]{12}", name) for name in names), (module, names)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG45 -- the builtins' namespace, exec and eval, type aliases and parameters
# ---------------------------------------------------------------------------
_OPENERS = (
    'x = __builtins__\n',
    'import builtins\n\nbuiltins.__dict__["chr"] = str\n',
    'import builtins as b\n\nvars(b).update(chr=str)\n',
    'exec("pass")\n',
    'eval("1")\n',
    '__import__("builtins").chr = str\n',
    'from builtins import __dict__ as d\n\nd["chr"] = str\n',
    'import sys\n\nsys._getframe().f_builtins["chr"] = str\n',
    'run = exec\n',
    'import builtins\n\ngetattr(builtins, "exec")("x = 1")\n',
    'm = __import__("builtins")\n\nm.__dict__["chr"] = str\n',
    'm = __import__("builtins")\n\nvars(m)["chr"] = str\n',
    'setattr(__import__("builtins"), "chr", str)\n',
)
_HARMLESS = ('import builtins\n\nx = __import__("os")\n\ny = x.getcwd()\nz = __import__("os").cpu_count()\n'
             'w = builtins.len("ab")\n')
_OPENED = 'import re\n\nP = r"a+"\n\n\ndef find(t):\n    return re.search(chr(98) + "+", t), re.search(P, t)\n'
_TYPED = ('import re\n\ntype chr = int\n\n\ndef find(t):\n    return re.search(chr(98), t)\n\n\n'
          'def typed[sorted](t):\n    return re.search("".join(sorted(("b", "a"))), t)\n')


def test_pg45_a_package_that_can_rebind_the_builtins_reads_none_as_itself(tmp_path):
    guard, restore = _load()
    try:
        alone = [pattern for _m, _l, _c, pattern, _f, _methods
                 in guard.sites(_tree(tmp_path / "alone", {"user.py": _OPENED}))]
        assert alone == ["b+", "a+"], ("control: nothing rebinds the builtins", alone)
        harmless = [pattern for module, _l, _c, pattern, _f, _methods
                    in guard.sites(_tree(tmp_path / "harmless", {"user.py": _OPENED, "opener.py": _HARMLESS}))
                    if module.endswith("user.py")]
        assert harmless == ["b+", "a+"], ("control: a module imported by name and only read opens nothing", harmless)
        for index, opener in enumerate(_OPENERS):
            rows = [pattern for module, _l, _c, pattern, _f, _methods
                    in guard.sites(_tree(tmp_path / f"o{index}", {"user.py": _OPENED, "opener.py": opener}))
                    if module.endswith("user.py")]
            assert rows == [None, "a+"], (opener, rows)
        own = _OPENED.replace("def find(t):", 'exec("pass")\n\n\ndef find(t):')
        rows = [pattern for _m, _l, _c, pattern, _f, _methods in guard.sites(_tree(tmp_path / "own", {"user.py": own}))]
        assert rows == [None, None], ("exec opens its own module's names", rows)
        if sys.version_info >= (3, 12):
            rows = [pattern for _m, _l, _c, pattern, _f, _methods
                    in guard.sites(_tree(tmp_path / "typed", {"typed.py": _TYPED}))]
            assert rows == [None, None], ("a type alias and a type parameter bind their names", rows)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG46 -- the ledger is empty (replaces PG44)
# ---------------------------------------------------------------------------
def test_pg46_the_ledger_is_empty_so_any_superlinear_pattern_fails_by_name(tmp_path):
    guard, restore = _load()
    try:
        assert guard.LEDGER == {}, guard.LEDGER
        code, lines = guard.run(_tree(tmp_path, {"marks.py": _QUADRATIC}))
        assert code == 1 and any("opti_oignon/marks.py: 1 superlinear pattern(s), 0 owed" in line for line in lines), (
            "witness: under the ledger as written, a superlinear pattern fails by name", lines)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG47 -- a branch that takes a run at once only where the real match fails
# ---------------------------------------------------------------------------
_PG47_SWALLOWED = 'import re\n\n\ndef strip(text):\n    return re.{0}(r"<[^>]+>(?P<real>)|<[^>{1}]*", {2}text)\n'


def test_pg47_a_branch_that_refuses_a_character_the_real_match_takes_is_found_reading_every_match(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = {}
        code, lines = guard.run(_tree(tmp_path, {"tags.py": _PG47_SWALLOWED.format("sub", "=", '"", ')}))
        assert code == 1 and any("tags.py: 1 superlinear" in line for line in lines), lines
        assert any("(<=)*" in line for line in lines), ("found by the unit of the head and the refused character", lines)
        assert ("<", "=", None) in guard.pumps(r"<[^>]+>(?P<real>)|<[^>=]*", 0)
        code, lines = guard.run(_tree(tmp_path / "taken", {"tags.py": _PG47_SWALLOWED.format("sub", "", '"", ')}))
        assert code == 0, ("control: a branch that takes every run at once reads every match in linear time", lines)
        code, lines = guard.run(_tree(tmp_path / "searched", {"tags.py": _PG47_SWALLOWED.format("search", "=", "")}))
        assert code == 0, ("searched only: one search stops at its first match", lines)
        for function in ("findall", "finditer", "split"):
            code, lines = guard.run(_tree(tmp_path / function, {"tags.py": _PG47_SWALLOWED.format(function, "=", "")}))
            assert code == 1 and any("(<=)*" in line for line in lines), (function, lines)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG48 -- a class of characters the fixed list does not show
# ---------------------------------------------------------------------------
_PG48_DIGITS = chr(0x660) + "-" + chr(0x669)


def test_pg48_a_class_no_character_of_the_fixed_list_shows_is_pumped_by_one_of_its_own():
    guard, restore = _load()
    try:
        pattern = r"^(?:\dy?|[" + _PG48_DIGITS + "a])+x"
        chars = [ch for _lead, ch, _body in guard.pumps(pattern, 0)]
        assert chars == ["0", chr(0x660), "a"], chars
        assert _steep(guard, pattern), "the digits both branches take, none of them in the fixed list"
        assert not _steep(guard, r"^(?:\dy?|[ba])+x"), "control: every class one branch takes"
        assert guard.first_chars(list(guard.sre_parse.parse(pattern)), 0) == ["0", "9", "a"], \
            "witness: the fixed list alone shows no such digit"
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG49 -- the classes of a pattern, read from the engine
# ---------------------------------------------------------------------------
def test_pg49_the_classes_of_a_pattern_are_read_from_the_engine_over_every_code_point():
    guard, restore = _load()
    try:
        import bisect
        import re
        import sys

        def classes(pattern, flags=0):
            parsed = guard.sre_parse.parse(pattern, flags)
            return guard.classes(list(parsed), parsed.state.flags)

        assert classes(r"ab\s*c") == [" ", "a", "0", "b", "c"]
        assert classes(r"k") == [" ", "k"] and classes(r"(?i)k") == [" ", "K"], "a literal with its cases"
        assert (0x212A, 0x212A) in guard.members("k", re.I | re.U), "the engine folds the Kelvin sign into k"
        assert classes(r"(?i:k)x") == [" ", "x", "K"], "a group reads under its own flags"
        assert classes(r"x") == [" ", "x"] and classes(r"\bx") == [" ", "a", "x"], "a word boundary reads words"
        assert classes(r"\Aa\Z") == [" ", "a"] and classes(r"^a$", re.M) == [" ", "\n", "a"], "a line anchor"
        assert classes(r"a$") == [" ", "\n", "a"], "an end that stands before a last newline"
        assert classes("[" + _PG48_DIGITS + "]") == [" ", chr(0x660)], "a class the list does not show"
        assert classes("[" + chr(0xD800) + "-" + chr(0xE005) + "]") == [" ", chr(0xE000)], "past the surrogates"
        assert classes("[" + chr(0xD800) + "-" + chr(0xDFFF) + "]") == [" ", chr(0xD800)], "surrogates only"
        assert classes(r"[^\s\S]") == [" "], "control: a class nothing enters adds none"
        assert guard.members(re.escape("x"), 0) == ((0x78, 0x78),), "a literal read with case is itself"
        for single in (r"\w", r"[^a-z]", ".", r"\d", "k"):
            for flags in (re.U, re.U | re.I, re.A, re.U | re.S):
                ranges = guard.members(single, flags)
                firsts = [first for first, _last in ranges]
                compiled = re.compile(single, flags)
                for point in range(0, sys.maxunicode + 1, 61):
                    at = bisect.bisect_right(firsts, point) - 1
                    inside = at >= 0 and ranges[at][1] >= point
                    assert inside is bool(compiled.fullmatch(chr(point))), (single, flags, point)
        pattern = r"(?i)[a-f\d]+\bk$"
        items = [re.compile(single, re.I) for single in (r"[a-f\d]", r"\w", "k", "\n")]
        shown = {tuple(bool(item.fullmatch(ch)) for item in items) for ch in classes(pattern)}
        assert len(shown) == len(classes(pattern)) > 3, "one character for each class, no two alike"
        for point in range(0, sys.maxunicode + 1, 97):
            assert tuple(bool(item.fullmatch(chr(point))) for item in items) in shown, point
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG50 -- the doubling, one character for each class (replaces PG19)
# ---------------------------------------------------------------------------
def test_pg50_the_doubling_reads_past_its_short_length_only_while_the_time_more_than_doubles():
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
        assert len(seen) == 12, ("one class of characters, twelve lengths, from 2 to 4,096", len(seen))
        seen.clear()
        result = guard.growth(r"[\s\d]*\s?\d?x", 0)
        assert len(seen) == 2 * 12, ("two classes the body takes, twelve lengths each", len(seen))
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


# ---------------------------------------------------------------------------
# PG51 -- every function of re, and the method it is read with (replaces PG21)
# ---------------------------------------------------------------------------
def test_pg51_every_function_of_re_gives_its_pattern_its_flags_and_the_method_it_is_read_with(tmp_path):
    guard, restore = _load()
    try:
        import re
        rows = [(line, pattern, flags, methods) for _m, line, _c, pattern, flags, methods
                in guard.sites(_tree(tmp_path, {"every.py": _EVERY}))]
        search, every = ("search",), ("findall",)
        assert rows == [(5, "a", int(re.I), every), (6, "b", int(re.M), search), (7, "c", int(re.S), ("match",)),
                        (8, "d", int(re.X), ("fullmatch",)), (9, "e", int(re.A), every), (10, "f", int(re.U), every),
                        (11, "g", int(re.I | re.M), every), (12, "h", int(re.S), every), (13, "i", int(re.X), every),
                        (14, "j", int(re.I), search), (15, "k", int(re.M), every), (16, "l", 0, search),
                        (17, "m", 0, every)], rows
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG52 -- the pumps of a pattern, one character for each class (replaces PG22)
# ---------------------------------------------------------------------------
def test_pg52_the_pumps_of_a_pattern_are_its_leads_the_classes_its_bodies_take_and_its_bodies():
    guard, restore = _load()
    try:
        import hashlib
        import re
        assert guard.pumps(r"ab\s*c", 0) == [("ab", " ", None)], "every space is one class here"
        assert guard.pumps(r"ab\s*\t?c", 0) == [("ab", " ", None), ("ab", "\t", None)], "a tab told apart"
        assert guard.pumps(r"x(?:<t>.)*y", re.S) == [("x", "<", "<t> ")]
        assert guard.pumps(r"[^\n]{2,}z", 0) == [("", " ", None), ("", "z", None)]
        assert guard.pumps(r"a{1,5}b", 0) == [], "a bounded repeat is not pumped"
        assert guard.pumps(r"(?=\s*x)y", 0)[0] == ("", " ", None), "a lookahead's repeat is pumped"
        assert guard._texts("ab", " ", "<t>", 8) == {"run": "ab" + " " * 8 + "\x00", "unit": "ab ab \x00",
                                                     "body": "ab<t><t>\x00"}
        assert guard._texts("", "a", None, 4) == {"run": "aaaa\x00"}
        assert guard._texts("abc", " ", "<tt>", 2) == {"run": "abc  \x00", "unit": "abc \x00", "body": "abc<tt>\x00"}
        assert guard._sample(list(guard.sre_parse.parse(r"(?:ab|c)\d{2}[xy]+")), 0) == "ab00x"
        assert guard._sample(list(guard.sre_parse.parse(r"(?-i:[A-Z])")), re.I) == "Z", "a group's own flags"
        assert guard.digest(r"\s*", 0) == hashlib.sha256(b"0:\\s*").hexdigest()[:12], "the ledger's names hold"
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG53 -- the classes the rest of the pattern refuses come first (replaces PG35)
# ---------------------------------------------------------------------------
def test_pg53_the_pump_characters_the_rest_of_the_pattern_refuses_come_first():
    guard, restore = _load()
    try:
        chars = [ch for lead, ch, body in guard.pumps(r"(\w+)+[a0_\xe9]", 0) if lead == ""]
        assert chars == ["b", "a"], chars
        assert _steep(guard, r"(\w+)+[a0_\xe9]"), "found with a character the end refuses"
        assert [ch for _l, ch, _b in guard.pumps(r"ab\s*\t?c", 0)] == [" ", "\t"], "the tab the rest takes, last"
        assert [ch for _l, ch, _b in guard.pumps(r"ab\s*\t?c", 0)] != [ch for _l, ch, _b in guard.pumps(r"ab\s*c", 0)], \
            "control: the rest of the pattern makes the order"
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG54 -- the forms the evaluator reads, sorted read sorted (replaces PG37)
# ---------------------------------------------------------------------------
_PG54_FORMS_ROWS = _FORMS_ROWS[:9] + [(33, '\\s*j', 0), (33, '\\s*k', 0)] + _FORMS_ROWS[11:]


def test_pg54_every_constant_form_gives_its_patterns_sorted_read_sorted_and_every_unread_form_is_dynamic(tmp_path):
    guard, restore = _load()
    try:
        assert _FORMS_ROWS[9:11] == [(33, '\\s*k', 0), (33, '\\s*j', 0)], "control: the rows the dict's order gave"
        rows = [(line, pattern, flags) for _m, line, _c, pattern, flags, _methods
                in guard.sites(_tree(tmp_path, {"forms.py": _FORMS}))]
        assert rows == _PG54_FORMS_ROWS, rows
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
        mixed = 'import re\n\n\ndef find(t):\n    for p in sorted(("b", 1)):\n        re.search(p, t)\n' \
                '    for q in sorted("ca"):\n        re.search(q, t)\n'
        rows = [(line, pattern) for _m, line, _c, pattern, _f, _methods
                in guard.sites(_tree(tmp_path / "mixed", {"mixed.py": mixed}))]
        assert rows == [(6, None), (8, "a"), (8, "c")], ("items that cannot be ordered are dynamic", rows)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG55 -- the pump helpers on a table of patterns (replaces PG39)
# ---------------------------------------------------------------------------
_PG55_PUMPS = {
    ("x[^\\n]*?y", 0): [("x", " ", None), ("x", "x", None), ("x", "y", None)],
    ("[a-c\\d]+z", 0): [("", "a", None)],
    ("[^a]+a", 0): [("", " ", None)],
    ("\\W+\\w", 0): [("", " ", None)],
    ("a.+?b", 16): [("a", " ", None), ("a", "a", None), ("a", "b", None)],
    ("^\\s+$", 8): [("", " ", None), ("", "\n", None)],
    ("\\d{2,}x", 0): [("", "0", None)],
    ("(\\w+)+[a0_\\xe9]", 0): [("", "b", None), ("", "a", None)],
}


def test_pg55_the_pump_helpers_give_their_exact_outputs_on_a_table_of_patterns():
    guard, restore = _load()
    try:
        assert len(_PUMP_TABLE) >= 20, "control: the table holds every kind of node"
        assert set(_PG55_PUMPS) <= {(row[0], row[1]) for row in _PUMP_TABLE}, "control: each new row is in the table"
        for pattern, flags, pumps, sample, first, groups, unbounded in _PUMP_TABLE:
            tree = list(guard.sre_parse.parse(pattern, flags))
            found = guard._groups(tree)
            assert guard.pumps(pattern, flags) == _PG55_PUMPS.get((pattern, flags), pumps), pattern
            assert guard._sample(tree, flags, found) == sample, pattern
            assert guard.first_chars(tree, flags, found)[:6] == first, pattern
            assert sorted(found) == groups, pattern
            assert guard.unbounded(pattern, flags) is unbounded, pattern
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG56 -- the costliest pump of a linear pattern (fake times, real loop; replaces PG40)
# ---------------------------------------------------------------------------
def test_pg56_a_linear_pattern_is_reported_by_its_costliest_pump_labelled_by_its_kind():
    guard, restore = _load()
    try:
        def timed(compiled, text, methods):
            return 1e-7 * len(text) * (2 if "\t" in text else 1) * (3 if text.count("ab") > 1 else 1)

        guard._timed = timed
        result = guard.growth(r"ab\s*\t?c", 0)
        assert not guard.superlinear(result), result
        assert result[2:] == ("(ab\t)*", "\t") and abs(result[0] - 2.0) < 1e-9, result
        assert abs(result[1] - 1e-7 * 4096 * 2 * 3) < 1e-12, ("the unit pump at 4,096", result)

        def mixed(compiled, text, methods):
            return 9.7e-6 * len(text) if "\t" in text else 2.5e-9 * len(text) ** 2

        guard._timed = mixed
        result = guard.growth(r"ab\s*\t?c", 0)
        assert guard.superlinear(result) and result[2:] == ("ab", " "), ("the first superlinear pump, at once", result)
        guard._timed = lambda compiled, text, methods: 1e-7 * len(text)
        result = guard.growth(r"ab\s*\t?c", 0)
        assert result[2:] == ("ab", " "), ("equal costs keep the first pump", result)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG57 -- a compiled name read with the costliest method its module calls
# ---------------------------------------------------------------------------
_PG57_NAMED = 'import re\n_TAG = re.compile(r"<[^>]+>(?P<real>)|<[^>=]*")\n\n\ndef read(text):\n    return {0}\n'


def test_pg57_a_compiled_name_is_read_with_the_costliest_method_its_module_calls_it_with(tmp_path):
    guard, restore = _load()
    try:
        search, every = ("search",), ("findall",)
        calls = (("_TAG.search(text)", search), ("_TAG.match(text)", ("match",)), ("_TAG.match(text, 3)", search),
                 ("_TAG.fullmatch(text) or _TAG.match(text)", ("fullmatch", "match")),
                 ("_TAG.match(text) or _TAG.search(text)", search), ('_TAG.sub("", text)', every),
                 ("_TAG.findall(text)", every), ("_TAG.finditer(text)", every), ("_TAG.split(text)", every),
                 ('_TAG.subn("", text)', every), ("_TAG.scanner(text)", every), ("[_TAG]", every),
                 ('_TAG.search(text) or _TAG.sub("", text)', every))
        for index, (call, methods) in enumerate(calls):
            rows = [found for _m, _l, _c, _p, _f, found
                    in guard.sites(_tree(tmp_path / str(index), {"tags.py": _PG57_NAMED.format(call)}))]
            assert rows == [methods], (call, rows)
        guard.LEDGER = {}
        code, lines = guard.run(_tree(tmp_path / "sub", {"tags.py": _PG57_NAMED.format('_TAG.sub("", text)')}))
        assert code == 1 and any("tags.py: 1 superlinear" in line for line in lines), lines
        code, lines = guard.run(_tree(tmp_path / "search", {"tags.py": _PG57_NAMED.format("_TAG.search(text)")}))
        assert code == 0, ("control: searched only, one search reads it in linear time", lines)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG58 -- a group's type flag takes the place of the pattern's
# ---------------------------------------------------------------------------
_PG58_TYPED = 'import re\n_WORD = re.compile({0!r}{1})\n\n\ndef read(text):\n    return _WORD.search(text)\n'


def test_pg58_a_group_s_type_flag_takes_the_place_of_the_pattern_s_as_the_engine_combines_them(tmp_path):
    guard, restore = _load()
    try:
        import itertools
        import re
        try:
            from re._compiler import _combine_flags
        except ImportError:  # Python 3.10
            from sre_compile import _combine_flags
        singles = (re.IGNORECASE, re.MULTILINE, re.DOTALL, re.VERBOSE, re.ASCII, re.LOCALE, re.UNICODE)
        every = [sum(chosen) for size in range(4) for chosen in itertools.combinations(singles, size)]
        for flags, added, removed in itertools.product(every, every, (0, re.IGNORECASE, re.IGNORECASE | re.DOTALL)):
            assert guard._local(flags, (None, added, removed, [])) == _combine_flags(flags, added, removed), (
                flags, added, removed)
        assert guard._local(re.UNICODE, (None, re.ASCII, 0, [])) == re.ASCII, "witness: the type flag is replaced"
        accent = chr(0xE9)
        run = "-" + accent + "*"
        guard.LEDGER = {}
        code, lines = guard.run(_tree(tmp_path / "ascii", {"words.py": _PG58_TYPED.format(run + r"(?a:\w)*y", "")}))
        assert code == 0, ("ASCII words inside the group: a run of accents splits once", lines)
        code, lines = guard.run(_tree(tmp_path / "unicode", {"words.py": _PG58_TYPED.format(
            run + r"(?u:\w)*y", ", re.A")}))
        assert code == 1 and any("words.py: 1 superlinear" in line for line in lines), (
            "Unicode words inside the group, under ASCII, share the accent", lines)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG59 -- a caret under multiline mode reads the newline as a class of its own
# ---------------------------------------------------------------------------
_PG59_CARET = 'import re\n_LINES = re.compile(r"(?m)^\\s*^\\s*x")\n\n\ndef read(text):\n    return _LINES.search(text)\n'


def test_pg59_a_caret_under_multiline_mode_reads_the_newline_as_a_class_of_its_own(tmp_path):
    guard, restore = _load()
    try:
        import re

        def classes(pattern, flags=0):
            parsed = guard.sre_parse.parse(pattern, flags)
            return guard.classes(list(parsed), parsed.state.flags)

        assert [av for op, av in guard.sre_parse.parse(r"(?m)^x") if op is guard.sre_constants.AT] == [
            guard.sre_constants.AT_BEGINNING], "witness: the parser writes a multiline caret as the text's start"
        assert "\n" in classes(r"(?m)^\s*^\s*x") and "\n" in classes(r"^\s*x", re.M)
        assert "\n" not in classes(r"^\s*x"), "control: outside multiline mode a caret is the text's start alone"
        guard.LEDGER = {}
        code, lines = guard.run(_tree(tmp_path, {"lines.py": _PG59_CARET}))
        assert code == 1 and any("lines.py: 1 superlinear" in line for line in lines), lines
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG60 -- a match call that unpacks its arguments is timed as a search
# ---------------------------------------------------------------------------
def test_pg60_a_match_call_that_unpacks_its_arguments_is_timed_as_a_search(tmp_path):
    guard, restore = _load()
    try:
        calls = (("_TAG.match(*args)", ("search",)), ("_TAG.match(text, **options)", ("search",)),
                 ("_TAG.fullmatch(*args)", ("search",)), ("_TAG.match(text)", ("match",)))
        for index, (call, methods) in enumerate(calls):
            rows = [found for _m, _l, _c, _p, _f, found
                    in guard.sites(_tree(tmp_path / str(index), {"tags.py": _PG57_NAMED.format(call)}))]
            assert rows == [methods], (call, rows)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG61 -- a lead before the repeat is written to pass an anchor or a lookaround in it
# ---------------------------------------------------------------------------
# (pattern, flags, call): an item before the repeat is written by its first character, a blank, which the anchor or
# lookaround after it refuses; only a newline passes.
_PG61_LEADS = [(r"\A\s^\s*\s*!", "re.M", "search"), (r"\s^\s*\s*!", "re.M", "match"),
               (r"(?m:\A\s^)\s*\s*!", "0", "search"), (r"\A(?=\n)\s\s*\s*!", "0", "search"),
               (r"\A\s(?<=\n)\s*\s*!", "0", "findall")]
_PG61_TEMPLATE = 'import re\n_LEAD = re.compile({0!r}, {1})\n\n\ndef read(text):\n    return _LEAD.{2}(text)\n'


def test_pg61_a_lead_before_the_repeat_is_written_to_pass_an_anchor_or_a_lookaround_in_it(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = {}
        for index, (pattern, flags, call) in enumerate(_PG61_LEADS):
            code, lines = guard.run(_tree(tmp_path / str(index), {"lead.py": _PG61_TEMPLATE.format(pattern, flags, call)}))
            assert code == 1 and any("lead.py: 1 superlinear" in line for line in lines), (pattern, lines)
        code, lines = guard.run(_tree(tmp_path / "control", {"lead.py": _PG61_TEMPLATE.format(
            r"\A\s\s*\s*!", "0", "search")}))
        assert code == 1, ("control: with no anchor in the lead, as before", lines)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG62 -- a pattern stored on an object and read in another module is timed the costliest way
# ---------------------------------------------------------------------------
_PG62_OWNER = ('import re\n\n\nclass Reader:\n    def __init__(self):\n        self.RX = re.compile(r" *x")\n\n'
               '    def first(self, text):\n        return self.RX.match(text)\n')
_PG62_OTHER = 'from .owner import Reader\n\n\ndef every(text):\n    return Reader().RX.findall(text)\n'


def test_pg62_a_pattern_stored_on_an_object_and_read_in_another_module_is_timed_the_costliest_way(tmp_path):
    guard, restore = _load()
    try:
        guard.LEDGER = {}
        code, lines = guard.run(_tree(tmp_path / "alone", {"owner.py": _PG62_OWNER}))
        assert code == 0, ("control: matched from the start in its own module only", lines)
        code, lines = guard.run(_tree(tmp_path / "read", {"owner.py": _PG62_OWNER, "other.py": _PG62_OTHER}))
        assert code == 1 and any("owner.py: 1 superlinear" in line for line in lines), lines
        held = _PG62_OWNER.replace("    def __init__(self):\n        self.RX = re.compile", "    RX = re.compile")
        assert held != _PG62_OWNER
        code, lines = guard.run(_tree(tmp_path / "class", {"owner.py": held, "other.py": _PG62_OTHER}))
        assert code == 1 and any("owner.py: 1 superlinear" in line for line in lines), ("a name in a class body", lines)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG63 -- a lead is written with the preferred class only where its item takes it, and before a word boundary too
# ---------------------------------------------------------------------------
# (pattern, flags, call): "#" refuses the newline the caret needs after the blank; a word boundary refuses the first
# character of a class that holds both a word character and another.
_PG63_LEADS = [(r"#\s^x*x*y", "re.M", "search"), (r"[a.]\bx*x*y", "0", "search"), (r"[ a]\Bx*x*y", "0", "findall")]


def test_pg63_a_lead_takes_the_preferred_class_only_where_its_item_does_and_before_a_word_boundary(tmp_path):
    guard, restore = _load()
    try:
        import re
        assert guard._sample(guard.sre_parse.parse("#"), 0, None, "\n") == "#", "an item that refuses it keeps its own"
        assert guard._sample(guard.sre_parse.parse(r"\s"), 0, None, "\n") == "\n"
        guard.LEDGER = {}
        for index, (pattern, flags, call) in enumerate(_PG63_LEADS):
            code, lines = guard.run(_tree(tmp_path / str(index), {"lead.py": _PG61_TEMPLATE.format(pattern, flags, call)}))
            assert code == 1 and any("lead.py: 1 superlinear" in line for line in lines), (pattern, lines)
        assert re.search(r"#\s^x*x*y", "#\n" + "x" * 50, re.M) is None, "witness: the lead the guard must write"
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG64 -- a lead writes the branch that holds the preferred class, and a class none of the candidates shows
# ---------------------------------------------------------------------------
_PG64_LEADS = [(r"^(?:ab|cd)(?<=d)\s*\s*$", "0", "search"), ("^[^\\x00-\\uffff]\\s*\\s*$", "0", "search")]


def test_pg64_a_lead_writes_the_branch_that_holds_the_preferred_class_and_a_class_no_candidate_shows(tmp_path):
    guard, restore = _load()
    try:
        assert guard._sample(guard.sre_parse.parse("(?:ab|cd)"), 0, None, "d") == "cd"
        assert guard._sample(guard.sre_parse.parse("(?:ab|cd|ed)"), 0, None, "d") == "cd", "the first that writes it"
        assert guard._sample(guard.sre_parse.parse("(?:ab|cd)"), 0, None, None) == "ab", "control: the first branch"
        astral = chr(0x10000)
        assert guard._sample(guard.sre_parse.parse("[^\\x00-\\uffff]"), 0, None, astral) == astral
        assert guard._sample(guard.sre_parse.parse("[^\\x00-\\uffff]"), 0, None, None) is None, "control: no candidate"
        guard.LEDGER = {}
        for index, (pattern, flags, call) in enumerate(_PG64_LEADS):
            code, lines = guard.run(_tree(tmp_path / str(index), {"lead.py": _PG61_TEMPLATE.format(pattern, flags, call)}))
            assert code == 1 and any("lead.py: 1 superlinear" in line for line in lines), (pattern, lines)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG65 -- a reading of the tree keeps none of the modules it parsed
# ---------------------------------------------------------------------------

def test_pg65_a_reading_of_the_tree_keeps_none_of_the_modules_it_parsed(tmp_path):
    guard, restore = _load()
    try:
        real, kept, pinned = guard._package, [], []

        def package(root):
            parsed = real(root)
            kept.extend(weakref.ref(module) for _, module in parsed if module is not None)
            pinned.append(parsed[0][1])
            return parsed

        guard._package = package
        root = _tree(tmp_path, {"a.py": _LINEAR, "b.py": "import re\n\n\ndef find(text):\n    return re.search(r'x+y', text)\n"})
        assert guard.sites(root), "the reading finds the patterns"
        gc.collect()
        assert len(kept) == 2 and [ref() for ref in kept if ref() is not None] == pinned, \
            "control: a module still held elsewhere is seen alive"
        pinned.clear()
        gc.collect()
        assert all(ref() is None for ref in kept), "no module the reading parsed is kept once it returns"
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG66 -- the least of two runs, on the thread's own clock (replaces PG20)
# ---------------------------------------------------------------------------
def test_pg66_each_time_is_the_least_of_two_runs_of_the_costliest_method_on_the_threads_own_clock():
    guard, restore = _load()
    try:
        ticks, calls = [0.0, 5.0, 10.0, 13.0, 20.0, 21.0, 30.0, 32.0], []

        class Clock:
            @staticmethod
            def thread_time():
                return ticks.pop(0)

            @staticmethod
            def perf_counter():
                raise AssertionError("the wall clock is never read")

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
# PG67 -- a reading counts the time its thread runs, never the time it waits
# ---------------------------------------------------------------------------
def test_pg67_a_reading_counts_the_time_its_thread_runs_never_the_time_it_waits():
    guard, restore = _load()
    try:
        import time

        class Waiting:
            def search(self, text):
                time.sleep(0.05)

        class Working:
            def search(self, text):
                end = time.thread_time() + 0.02
                while time.thread_time() < end:
                    pass

        assert guard._timed(Waiting(), "x", ("search",)) < 0.01, "a wait is not the pattern's time"
        assert guard._timed(Working(), "x", ("search",)) >= 0.015, "control: work is"
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG68 -- an exponential pattern, within its thread's time (replaces PG34)
# ---------------------------------------------------------------------------
def test_pg68_an_exponential_pattern_ends_its_pump_at_its_first_burst_and_is_found_within_its_threads_time():
    guard, restore = _load()
    try:
        import time
        for pattern in (r"(a+)+b", r"(?:a+){2,9}b", r"(?:x|xx)+y", r"(?:\w+\s?)+:"):
            start = time.thread_time()
            result = _steep(guard, pattern)
            assert result is not None and result[0] >= guard.BURST, (pattern, result)
            assert time.thread_time() - start < 2.0, (pattern, "never hangs")
    finally:
        restore()


# ---------------------------------------------------------------------------
# PG69 -- a growth read under the bar, over the floor, reads its shorter length again
# ---------------------------------------------------------------------------
def test_pg69_a_growth_read_under_the_bar_over_the_floor_reads_its_shorter_length_again():
    guard, restore = _load()
    try:
        seen = []

        def spell(cost, slow, slowed=(0,), factor=3.0):
            """Fake times: ``cost`` of each length; the readings of the length ``slow`` whose rank is in
            ``slowed`` taken ``factor`` times slower, as in a slow spell of the machine."""
            ranks = {}

            def timed(compiled, text, methods):
                size = len(text)
                seen.append(size)
                rank = ranks[size] = ranks.get(size, -1) + 1
                return cost(size) * (factor if size == slow and rank in slowed else 1.0)
            return timed

        def quadratic(size):
            return guard.STOP * (size / 4097) ** 2

        guard._timed = spell(quadratic, 1025)
        result = guard.growth(r"\s*x", 0)
        assert guard.superlinear(result) and result[0] > 3.9, ("a slow spell in the shorter reading", result)
        assert seen[-4:] == [2049, 4097, 1025, 1025], ("the shorter length read twice more", seen[-5:])
        seen.clear()
        guard._timed = spell(quadratic, 1025, slowed=(0, 2))
        result = guard.growth(r"\s*x", 0)
        assert guard.superlinear(result) and result[0] > 3.9, ("its least time, not its last", result)
        seen.clear()
        guard._timed = spell(quadratic, 1025, factor=2.0)
        result = guard.growth(r"\s*x", 0)
        assert guard.superlinear(result) and result[0] > 3.9, ("a dip between the hint and the bar", result)
        seen.clear()
        guard._timed = spell(lambda size: guard.FLOOR * (size / 16385) ** 2, 4097)
        result = guard.growth(r"\s*x", 0)
        assert guard.superlinear(result) and result[1] == guard.FLOOR and seen[-4:] == [8193, 16385, 4097, 4097], (
            "the floor itself is read again", result, seen[-5:])
        seen.clear()

        def zeroed(compiled, text, methods):
            size = len(text)
            seen.append(size)
            if size == 1025 and seen.count(1025) > 1:
                return 0.0
            return quadratic(size) * (3.0 if size == 1025 else 1.0)

        guard._timed = zeroed
        result = guard.growth(r"\s*x", 0)
        assert not guard.superlinear(result) and result[0] == 1.0, ("a reading again of 0 divides nothing", result)
        seen.clear()
        guard._timed = spell(lambda size: 0.004 * (size / 16385) ** 2, 4097)
        result = guard.growth(r"\s*x", 0)
        assert guard.superlinear(result) and result[0] > 3.9 and seen[-4:] == [8193, 16385, 4097, 4097], (
            "over the floor at the ceiling, under the stop", result, seen[-5:])
        seen.clear()
        guard._timed = spell(quadratic, None)
        assert guard.superlinear(guard.growth(r"\s*x", 0)) and len(seen) == 12, (
            "control: a growth read over the bar is not read again", seen)
        seen.clear()
        guard._timed = spell(lambda size: 3.0 ** ((size - 1).bit_length() - 13) * 2.0 ** -10, None)
        result = guard.growth(r"\s*x", 0)
        assert result[0] == guard.GROWTH and guard.superlinear(result) and len(seen) == 14, (
            "a growth read at the bar exactly is not read again", result, seen[-4:])
        seen.clear()
        guard._timed = spell(lambda size: 1e-7 * size, None)
        result = guard.growth(r"\s*x", 0)
        assert not guard.superlinear(result) and len(seen) == 12, ("under the floor nothing is read again", seen)
    finally:
        restore()
