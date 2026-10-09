#!/usr/bin/env python3
"""Contracts for the guard that censuses every write into a store a model reads back.

A model's context is assembled from stores -- facts, notes, skills, the Core,
the transcript, the caches, the projects and their documents, the prompt
templates -- and a write into one of them is where the platform decides what a
model will be told later. The gates on those writes protect only the writes
that go through them. The write census guard finds every write site of every
store it knows, follows a store object through the bindings of one module and
across modules, and holds each site housed, gated by dominance, exempt with a
reason a predicate checks where one can, or owed in a sealed ledger that may
only shrink; and it holds the list of stores closed: a module that opens a
database is a store's house or is outside by name.

  * SW1 -- capable on the real tree: the writes the platform is known to make
    are found, the ones only a cross-module binding reaches among them.
  * SW2 -- exact on a test package: every binding form within a module is
    followed, and a dictionary's ``update``, prose and a test of a name are
    not sites.
  * SW3 -- a pair nobody classifies is a violation by name.
  * SW4 -- an owed pair is tolerated while its module has not moved.
  * SW5 -- an owed module that moved, or grew, is refused.
  * SW6 -- paying the debt makes the entry stale.
  * SW7 -- a site of a gated pair with no gate before it is refused; with the
    gate before it, it is not.
  * SW8 -- an exemption whose module has no site, or is gone, is stale.
  * SW9 -- a module that cannot be read, listed or parsed fails the census by
    name.
  * SW10 -- the real tree satisfies its own guard, and the green line carries
    its denominators.
  * SW11 -- the entry point refuses a violation and an empty estate.
  * SW12 -- every seal is a full digest of the module's text.
  * SW13 -- a store object is followed across modules: an argument to another
    module's function, a carrier object, an argument to an imported class's
    method, a dependency provider, an exported module-level object.
  * SW14 -- a gate counts by dominance, never by line order: an early exit, a
    guarding condition, a verdict bound once, a closure defined after the
    verdict, a raising gate, an approval, a filter count; a gate in a branch, a
    verdict ignored, a gate after the site or inside a ``try``, a verdict
    rebound, do not.
  * SW15 -- a decision gate covers its own body, and a function is gated by
    its callers only when every reference to it is a call from a gated
    position, across modules.
  * SW16 -- each checked exemption is refused site by site when its predicate
    fails: route, unread, keyed, quiet, script, instance.
  * SW17 -- a gate names contracts that exist in a suite that names its home;
    a gate or an exemption kind no table defines, or a gate its home does not
    define, is refused.
  * SW18 -- a store whose house does not define what the table names is
    refused, and so is a house gate a write method does not call first.
  * SW19 -- a pair classified twice is refused.
  * SW20 -- the list of stores is closed: a module that opens a database and
    is no house, no helper and not outside by name is refused, and an outside
    entry that opens nothing is stale.
  * SW21 -- two censuses agree on the real tree: every call of a write method
    with a distinctive name, or every ``getattr`` that names one, is a census
    site, a call a store makes on itself, or one of the calls named here with
    the reason it writes no census store.

The first review found what the census could not see, and each answer has
its contract, the forms widening SW2, SW13 to SW18, SW20 and SW21 besides:

  * SW22 -- a module looked up by a constant name (``sys.modules``,
    ``import_module``) and a dispatch table of local functions are followed.
  * SW23 -- a gate whose authority comes from its caller -- an approval, an
    acceptance -- holds only when a route handler or a named entry reaches it.
  * SW24 -- a private member of a store reached outside its house is a site;
    inside the house, and a dunder, it is not.

The second review found what the first answers still let through:

  * SW25 -- every form of a module lookup and of a dispatch table is
    followed, a name bound to two modules names both, and a function
    defined under a module-level ``try`` receives what a caller hands it.
  * SW26 -- the callers rule sees a writer a decision returns, a parameter
    that may be the module, a lazy export, a method's class-body alias, and
    does not take another module's function of the same name for a caller.
  * SW27 -- an authority reaches through a helper, and a house gate covers
    its write methods, never a private member that goes around them.
  * SW28 -- a reason in prose covers the sites it lists by function and
    method, a site swapped for another included, and a constant -- a
    literal, a container, an f-string, a module constant -- binds a write to
    no context.
  * SW29 -- a gate holds in the ``else`` of its refusal and past an ``else``
    that leaves, never through a nested scope that rebinds or shadows it.
  * SW30 -- a proof is a test that runs (not ignored, deselected, skipped,
    nor outside a ``Test`` class, nor named in a docstring), and a nesting
    too deep to follow fails the census by name.

The third review found what the second answers still let through, and a
result that hung on the hash seed:

  * SW31 -- a receiver a parameter, a lambda, a loop, a second assignment or
    an attribute bound to a module may hold is no store object: the callers
    rule counts its call.
  * SW32 -- a decision hands out no writer: a partial it returns or keeps, a
    nested function, a lambda or a generator is refused; a partial it calls
    at once, or through a name of its own body, is its own effect.
  * SW33 -- a verdict covers no nested function or lambda after it, which run
    whenever called; a filter covers its data through a closure, unless a
    lambda's parameter shadows it.
  * SW34 -- an instance is placed by a closed grammar, through names bound to
    nothing else: a place that merely holds a temporary path, the shared
    temporary root, a parent step, a positional place or a subclass is
    refused.
  * SW35 -- an expression names every module it may, whichever is bound
    first, through ``or``, a pair of imports in a ``try``, or an attribute.
  * SW36 -- a gate is its own name, bound once, or its home's attribute; a
    rebinding, a lambda or a pattern capture is no gate.
  * SW37 -- a module imported by keyword, by a list of names that is no
    literal, or reached as an attribute of its package is followed.
  * SW38 -- a dispatch table that holds bound methods and functions of other
    modules is followed.
  * SW39 -- a proof is read through classes, ``pytestmark``, a class pytest
    does not collect, a suite that skips itself, and a selection rule that
    deselects by class or selects by an expression.
  * SW40 -- a house gate covers its methods on store objects, not a house
    function of the same name, and binds every override in a subclass.
  * SW41 -- a house function defined under a module-level ``try`` is found.
  * SW42 -- a site another gate of its pair covers needs no authority.

The fourth review, two readers in parallel, found what the third answers
still let through, and one live debt (a cache keyed to a module constant):

  * SW43 -- a house gate holds through every subclass (no rebound gate, no
    base ahead, no ``super(Base, self)``) and only on store objects.
  * SW44 -- a route is a handler of a router the module builds; a closure it
    hands on, or a registry's ``post``, is no route.
  * SW45 -- a receiver that may be the module (a conditional, a container, a
    chooser, a table, a held attribute) is no store object.
  * SW46 -- a coroutine or a generator counts only where it is consumed.
  * SW47 -- a header runs around its scope; a lambda's own names stand in
    front of a filtered one; a closure may check the verdict around it.
  * SW48 -- a star import may rebind any name the module reads.
  * SW49 -- a key is live only when the function that writes computes it.
  * SW50 -- a module is read by its syntax tree, never its text.
  * SW51 -- a proof is read as pytest reads the selection: prefixes, paths,
    bases, ``del``, ``if``, mark aliases, raises, async, conftest hooks.
  * SW52 -- a name names every definition and every object it may.
  * SW53 -- a class table read through ``self`` and an inherited method in a
    table are followed.
  * SW54 -- rewiring a store object outside its house is a site and unmakes
    an own object.
  * SW55 -- ``getattr`` of a package, a starred ``fromlist``, a re-exported
    alias, a base read by ``getattr``, a lookup by name are followed.
  * SW56 -- a gate or a decision defined twice answers for nothing; imports
    that all lead to the gate answer for it; a registry's decorator is no
    method marker.
  * SW57 -- a program run by ``runpy``, a temporary directory object and a
    ``TYPE_CHECKING`` stub are read for what they are.

The fifth review, and the sixth with a metamorphic fuzzing of the census --
fourteen thousand meaning-preserving rewrites of planted writes -- found what
the fourth answers still let through. Each answer widens a contract: SW2 (a
dependency, a context manager or a generator that yields; an element through
``next``, ``enumerate``, ``zip``, ``sorted`` or a literal's ``items``; a
class's own factory, ``super()``, an object called; an annotated dependency,
a provider as a default, ``*args`` and ``**kwargs``, a match capture,
``setattr``, ``getattr`` called; a function handed with its arguments to a
task, a thread or ``map``; an injecting decorator; a dataclass and a named
tuple, across modules too; a star import's function), SW16 (a read by
``getattr``), SW18 (SQL anywhere, in a constant, a write reached by name, a
private helper, the house's own class), SW20 (a database imported by name, a
helper of any stem), SW31, SW43 to SW49, SW51 (exits, the tests' folder's
configuration), SW52 (re-exports that branch at every step, each place
visited once), SW54 to SW57.

The guard is loaded alone in an isolation window, and every test package is
written in memory: no test reaches the maintainer's data. The real tree is
censused once per session and shared by the contracts that read it.
"""

import ast
import hashlib
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate  # noqa: E402

_GUARD = REPO / ".github" / "scripts" / "write_census_guard.py"

BUDGET_S = {
    "test_sw1_the_census_finds_the_writes_the_platform_is_known_to_make": 12.0,
    "test_sw2_every_binding_form_is_followed_and_nothing_else_is_a_site": 2.0,
    "test_sw3_a_pair_nobody_classifies_is_a_violation_by_name": 2.0,
    "test_sw4_an_owed_pair_is_tolerated_while_its_module_has_not_moved": 2.0,
    "test_sw5_an_owed_module_that_moved_or_grew_is_refused": 2.0,
    "test_sw6_paying_the_debt_makes_the_entry_stale": 2.0,
    "test_sw7_a_gated_site_needs_its_gate_before_it": 2.0,
    "test_sw8_an_exemption_with_no_site_is_stale": 2.0,
    "test_sw9_what_cannot_be_read_listed_or_parsed_fails_by_name": 2.0,
    "test_sw10_the_real_tree_satisfies_its_own_guard_with_its_denominators": 12.0,
    "test_sw11_the_entry_point_refuses_a_violation_and_an_empty_estate": 2.0,
    "test_sw12_every_seal_is_a_full_digest_of_its_module": 12.0,
    "test_sw13_a_store_object_is_followed_across_modules": 2.0,
    "test_sw14_a_gate_counts_by_dominance_never_by_line_order": 2.0,
    "test_sw15_a_function_is_gated_by_its_callers_only_when_every_caller_is": 2.0,
    "test_sw16_each_checked_exemption_is_refused_where_its_predicate_fails": 2.0,
    "test_sw17_a_gate_is_defined_at_home_and_proven_by_a_suite_that_names_it": 2.0,
    "test_sw18_a_store_the_table_misdescribes_is_refused": 2.0,
    "test_sw19_a_pair_classified_twice_is_refused": 2.0,
    "test_sw20_every_module_that_opens_a_database_is_classified": 2.0,
    "test_sw21_a_textual_census_of_write_names_agrees_with_the_binding_census": 12.0,
    "test_sw22_a_module_looked_up_by_name_and_a_dispatch_table_are_followed": 2.0,
    "test_sw23_a_gate_whose_authority_is_the_callers_is_reached_only_from_the_users_gesture": 2.0,
    "test_sw24_a_private_member_of_a_store_reached_outside_its_house_is_a_site": 2.0,
    "test_sw25_every_form_of_a_module_lookup_and_a_dispatch_table_is_followed": 2.0,
    "test_sw26_the_callers_rule_follows_every_reference_and_only_those": 2.0,
    "test_sw27_an_authority_reaches_through_helpers_and_a_house_covers_only_its_writes": 2.0,
    "test_sw28_a_reason_in_prose_lists_its_sites_and_a_constant_binds_no_context": 2.0,
    "test_sw29_a_gate_holds_in_its_else_and_never_through_a_rebinding_scope": 2.0,
    "test_sw30_a_proof_must_run_and_a_nesting_too_deep_fails_by_name": 2.0,
    "test_sw31_a_receiver_bound_by_anything_but_an_import_may_be_the_module": 2.0,
    "test_sw32_a_decision_hands_out_no_writer": 2.0,
    "test_sw33_a_verdict_covers_no_nested_function_or_lambda_and_a_filter_follows_its_data": 2.0,
    "test_sw34_an_instance_is_placed_by_a_closed_grammar_through_names_bound_to_nothing_else": 2.0,
    "test_sw35_an_expression_names_every_module_it_may": 2.0,
    "test_sw36_a_gate_is_its_own_name_bound_once_or_its_homes_attribute": 2.0,
    "test_sw37_a_module_imported_by_keyword_by_a_list_of_names_or_through_its_package_is_followed": 2.0,
    "test_sw38_a_dispatch_table_of_bound_methods_and_other_modules_functions_is_followed": 2.0,
    "test_sw39_a_proof_is_read_through_classes_marks_and_the_selection_rule": 2.0,
    "test_sw40_a_house_gate_covers_its_methods_on_store_objects_and_binds_their_overrides": 2.0,
    "test_sw41_a_house_function_defined_under_a_module_level_try_is_found": 2.0,
    "test_sw42_a_site_another_gate_of_its_pair_covers_needs_no_authority": 2.0,
    "test_sw43_a_house_gate_holds_through_every_subclass_and_only_on_store_objects": 2.0,
    "test_sw44_a_route_is_a_routers_handler_and_its_closure_is_not": 2.0,
    "test_sw45_a_receiver_that_may_be_the_module_is_no_store_object": 2.0,
    "test_sw46_a_coroutine_or_a_generator_counts_only_where_it_is_consumed": 2.0,
    "test_sw47_a_header_runs_around_its_scope_and_a_lambdas_own_names_stand_in_front": 2.0,
    "test_sw48_a_star_import_may_rebind_any_name_the_module_reads": 2.0,
    "test_sw49_a_key_is_live_only_when_the_function_that_writes_computes_it": 2.0,
    "test_sw50_a_module_is_read_by_its_syntax_tree_never_its_text": 2.0,
    "test_sw51_a_proof_is_read_as_pytest_reads_the_selection": 2.0,
    "test_sw52_a_name_names_every_definition_and_every_object_it_may": 2.0,
    "test_sw53_a_class_table_and_an_inherited_method_in_a_table_are_followed": 2.0,
    "test_sw54_rewiring_a_store_object_outside_its_house_is_a_site_and_unmakes_an_own_object": 2.0,
    "test_sw55_a_package_attribute_a_starred_fromlist_a_lookup_and_a_getattr_are_followed": 2.0,
    "test_sw56_a_gate_defined_twice_answers_for_nothing_and_its_imports_answer_for_it": 2.0,
    "test_sw57_a_program_run_by_runpy_a_temporary_object_and_a_type_checking_stub_are_read_for_what_they_are": 2.0,
}


def _load():
    loaded, restore = isolate(targets={"write_census_guard": _GUARD})
    return loaded["write_census_guard"], restore


# ---------------------------------------------------------------------------
# A test store and its house, written in memory.
# ---------------------------------------------------------------------------
_JOT = "opti_oignon/jot.py"
_JOT_TEXT = '''"""A test store."""


class Jot:
    def __init__(self, path=None):
        self.path = path

    def put(self, text, context=None):
        return text

    def drop(self, key):
        return key

    def read_back(self):
        return []


def get_jot():
    return Jot()


jot = Jot()


def put_line(text):
    jot.put(text)
'''


def _jot(guard, **changes):
    fields = dict(house=_JOT, classes=("Jot",), accesses=("get_jot",), instances=("jot",), writes=("put", "drop"),
                  functions=("put_line",), quiet=("drop",), reads=("read_back",), keys=("context",), place="path",
                  backs=(), reason="a test store")
    fields.update(changes)
    return guard.Store(**fields)


def _tables(guard, *, stores=None, gates=None, gated=None, exempt=None, ledger=None, outside=None, helpers=None,
            argued=None):
    guard.STORES = stores if stores is not None else {"jot": _jot(guard)}
    guard.GATES = gates or {}
    guard.GATED = gated or {}
    guard.EXEMPT = exempt or {}
    guard.LEDGER = ledger or {}
    guard.OUTSIDE = outside or {}
    guard.HELPERS = helpers or {}
    guard.ARGUED_SITES = argued or {}


def _census(guard, modules, root=None):
    estate = guard.Estate(Path(root) if root is not None else Path("/nonexistent-write-census"),
                          {"opti_oignon/__init__.py": "", _JOT: _JOT_TEXT, **modules}, [])
    return guard.take_census(estate)


def _sites(census):
    return {rel: sorted((s.store, s.method, s.function) for s in sites) for rel, sites in census.sites().items()}


def _line(text, needle):
    for number, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return number
    raise AssertionError(f"{needle!r} not in the test module")


_REAL = []


def _real(guard):
    """The census of the repository, taken once per session by the first contract that reads it."""
    if not _REAL:
        _REAL.append(guard.run(REPO))
    return _REAL[0]


# ---------------------------------------------------------------------------
# SW1
# ---------------------------------------------------------------------------
def test_sw1_the_census_finds_the_writes_the_platform_is_known_to_make():
    guard, restore = _load()
    try:
        result = _real(guard)
    finally:
        restore()
    sites = {rel: [(s.store, s.method, s.function) for s in found] for rel, found in result.census.sites().items()}
    # The one write path of the review queue: three writes into the facts,
    # three into the notes, all in apply_write.
    assert sorted(sites.get("opti_oignon/pending_writes.py", [])) == sorted([
        ("facts", "add", "apply_write"), ("facts", "update", "apply_write"),
        ("facts", "soft_delete", "apply_write"), ("notes", "add_note", "apply_write"),
        ("notes", "update_note", "apply_write"), ("notes", "delete_note", "apply_write"),
    ]), sites.get("opti_oignon/pending_writes.py")
    # The capture and the manual extraction call the extractor's write function.
    assert ("extraction", "extract_and_store", "_default_runner._job") in sites.get(
        "opti_oignon/memory/auto_capture.py", []), sites.get("opti_oignon/memory/auto_capture.py")
    assert ("extraction", "_extract_and_store", "extract_facts") in sites.get(
        "opti_oignon/api/routes_memory.py", []), sites.get("opti_oignon/api/routes_memory.py")
    # Reached only across modules: the caption's store comes from the route's
    # dependency, and the onion's Core from the librarian's state object.
    assert ("notes", "update_attachment", "caption_attachment") in sites.get("opti_oignon/notes/caption.py", [])
    assert sorted(sites.get("opti_oignon/memory/onion_store.py", [])) == [
        ("cellar", "store", "_rebuild"), ("core", "add", "_rebuild"), ("core", "supersede", "_rebuild"),
        ("flesh", "append", "_rebuild"), ("peels", "add", "_rebuild"), ("receipts", "append", "_rebuild"),
    ], sites.get("opti_oignon/memory/onion_store.py")
    # Reached only through a constant module lookup and two getattr: the
    # synced conversation lands in the transcript.
    assert ("conversation", "apply_synced_conversation", "_default_conversation_sink") in sites.get(
        "opti_oignon/veilid/sync_engine.py", []), sites.get("opti_oignon/veilid/sync_engine.py")
    # The probe can count: well over a hundred sites in dozens of modules.
    total = sum(len(found) for found in sites.values())
    assert total > 100 and len(sites) > 30, (total, len(sites))


# ---------------------------------------------------------------------------
# SW2
# ---------------------------------------------------------------------------
_FORMS = {
    "opti_oignon/f_plain.py": "from opti_oignon.jot import get_jot\n\n\ndef f():\n    get_jot().put('x')\n",
    "opti_oignon/f_alias.py": "from opti_oignon.jot import get_jot as g\n\n\ndef f():\n    s = g()\n    s.put('x')\n",
    "opti_oignon/sub/__init__.py": "",
    "opti_oignon/sub/f_relative.py": "from ..jot import jot\n\n\ndef f():\n    jot.put('x')\n",
    "opti_oignon/f_local_import.py": "def f():\n    from opti_oignon.jot import get_jot\n    get_jot().put('x')\n",
    "opti_oignon/f_try.py": (
        "try:\n    from opti_oignon.jot import jot as _j\nexcept ImportError:\n    _j = None\n\n\n"
        "def f():\n    _j.put('x')\n"),
    "opti_oignon/f_returner.py": (
        "from opti_oignon.jot import get_jot\n\n\ndef _require():\n    return get_jot()\n\n\n"
        "def f():\n    _require().put('x')\n"),
    "opti_oignon/f_method.py": (
        "from opti_oignon.jot import get_jot\n\n\nclass E:\n    def __init__(self):\n        self._store = get_jot()\n\n"
        "    def _get(self):\n        return self._store\n\n    def f(self):\n        self._get().put('x')\n"),
    "opti_oignon/f_attribute.py": (
        "from opti_oignon.jot import get_jot\n\n\nclass E:\n    def __init__(self):\n        self.s = get_jot()\n\n"
        "    def f(self):\n        self.s.put('x')\n"),
    "opti_oignon/f_argument.py": (
        "from opti_oignon.jot import get_jot\n\n\ndef w(store, x):\n    store.put(x)\n\n\n"
        "def d():\n    w(get_jot(), 'x')\n"),
    "opti_oignon/f_instance.py": "from opti_oignon.jot import jot\n\n\ndef f():\n    jot.drop('k')\n",
    "opti_oignon/f_sink.py": "from opti_oignon.jot import put_line\n\n\ndef f():\n    put_line('x')\n",
    "opti_oignon/f_handed.py": "from opti_oignon.jot import get_jot\n\n\ndef f():\n    return get_jot().put\n",
    "opti_oignon/f_getattr.py": (
        "from opti_oignon.jot import get_jot\n\n\ndef f(name):\n    getattr(get_jot(), 'put')('x')\n"
        "    getattr(get_jot(), name)('x')\n    getattr(get_jot(), 'read_back')()\n"),
    "opti_oignon/f_dotted.py": "import opti_oignon.jot\n\n\ndef f():\n    opti_oignon.jot.get_jot().put('x')\n",
    "opti_oignon/f_container.py": (
        "from opti_oignon.jot import get_jot\n\nstores = {}\n\n\ndef f():\n    stores['a'] = get_jot()\n"
        "    stores.get('a').put('x')\n"),
    "opti_oignon/f_property.py": (
        "from opti_oignon.jot import get_jot\n\n\nclass P:\n    @property\n    def store(self):\n"
        "        return get_jot()\n\n    def f(self):\n        self.store.put('x')\n"),
    "opti_oignon/f_subclass.py": (
        "from opti_oignon.jot import Jot\n\n\nclass MyJot(Jot):\n    def save(self):\n        self.put('x')\n"),
    "opti_oignon/f_private.py": "from opti_oignon.jot import get_jot\n\n\ndef f():\n    return get_jot()._rows\n",
    # Not sites: a dictionary's update and a queue's put in a module that
    # imports the store, prose, a test of a name, a module that never imports it.
    "opti_oignon/n_other.py": (
        "from opti_oignon.jot import get_jot\n\n\ndef f(q):\n    d = {}\n    d.update({'a': 1})\n    q.put(1)\n"
        "    return get_jot\n"),
    "opti_oignon/n_prose.py": (
        '"""Calls get_jot().put(x) and put_line(x)."""\nfrom opti_oignon.jot import get_jot, put_line\n\n\n'
        "def f():\n    # get_jot().put('x')\n    return 'put_line(x)'\n"),
    "opti_oignon/n_tested.py": (
        "from opti_oignon.jot import put_line\n\n\ndef f():\n    if put_line is None or not put_line:\n        return 0\n"
        "    return 1\n"),
    "opti_oignon/n_unrelated.py": "def f(store):\n    store.put('x')\n    store.drop('y')\n",
}


# Binding forms the sixth review and the fuzzing found, each followed to its write.
_FORMS_SIX = {
    "opti_oignon/s_yield.py": (
        "from fastapi import Depends\n\nfrom opti_oignon.jot import get_jot\n\n\ndef get_db():\n    db = get_jot()\n"
        "    try:\n        yield db\n    finally:\n        pass\n\n\ndef f(store=Depends(get_db)):\n    store.put('x')\n"),
    "opti_oignon/s_context.py": (
        "from contextlib import contextmanager\n\nfrom opti_oignon.jot import get_jot\n\n\n@contextmanager\n"
        "def opened():\n    yield get_jot()\n\n\ndef f():\n    with opened() as s:\n        s.put('x')\n"),
    "opti_oignon/s_generator.py": (
        "from opti_oignon.jot import get_jot\n\n\ndef gen():\n    yield get_jot()\n\n\ndef f():\n    for s in gen():\n"
        "        s.put('x')\n"),
    "opti_oignon/s_next.py": "from opti_oignon.jot import get_jot\n\n\ndef f():\n    next(iter([get_jot()])).put('x')\n",
    "opti_oignon/s_enumerate.py": (
        "from opti_oignon.jot import get_jot\n\nSTORES = [get_jot()]\n\n\ndef f():\n    for i, s in enumerate(STORES):\n"
        "        s.put(i)\n"),
    "opti_oignon/s_zip.py": (
        "from opti_oignon.jot import get_jot\n\n\ndef f(names):\n    for s, n in zip([get_jot()], names):\n"
        "        s.put(n)\n"),
    "opti_oignon/s_sorted.py": "from opti_oignon.jot import get_jot\n\n\ndef f():\n    sorted([get_jot()], key=id)[0].put('x')\n",
    "opti_oignon/s_items.py": (
        "from opti_oignon.jot import get_jot\n\n\ndef f():\n    for k, s in {'a': get_jot()}.items():\n        s.put(k)\n"),
    "opti_oignon/s_partial.py": (
        "from functools import partial\n\nfrom opti_oignon.jot import get_jot\n\n\ndef f():\n    partial(get_jot)().put('x')\n"),
    "opti_oignon/s_factory.py": (
        "from opti_oignon.jot import Jot\n\n\nclass Made(Jot):\n    @classmethod\n    def make(cls):\n        return cls()\n\n\n"
        "def f():\n    Made.make().put('x')\n"),
    "opti_oignon/s_super.py": (
        "from opti_oignon.jot import Jot\n\n\nclass Sup(Jot):\n    def save(self, x):\n        return super().put(x)\n"),
    "opti_oignon/s_called.py": (
        "from opti_oignon.jot import get_jot\n\n\nclass P:\n    def __call__(self):\n        return get_jot()\n\n\n"
        "def f():\n    P()().put('x')\n"),
    "opti_oignon/s_annotated.py": (
        "from typing import Annotated\n\nfrom fastapi import Depends\n\nfrom opti_oignon.jot import get_jot\n\n\n"
        "def f(store: Annotated[object, Depends(get_jot)]):\n    store.put('x')\n"),
    "opti_oignon/s_default.py": "from opti_oignon.jot import get_jot\n\n\ndef f(maker=get_jot):\n    maker().put('x')\n",
    "opti_oignon/s_varargs.py": (
        "from opti_oignon.jot import get_jot\n\n\ndef w(*args, **kwargs):\n    args[0].put('x')\n    kwargs['s'].drop('k')\n\n\n"
        "def d():\n    w(get_jot(), s=get_jot())\n"),
    "opti_oignon/s_match.py": (
        "from opti_oignon.jot import get_jot\n\n\ndef f(x):\n    match get_jot():\n        case s:\n            s.put(x)\n"),
    "opti_oignon/s_setattr.py": (
        "from opti_oignon.jot import get_jot\n\n\nclass H:\n    def __init__(self):\n        setattr(self, 'store', get_jot())\n\n"
        "    def f(self):\n        self.store.put('x')\n"),
    "opti_oignon/s_getattr_call.py": (
        "from opti_oignon.jot import get_jot\n\n\nclass K:\n    def run(self, store):\n        store.drop('k')\n\n\n"
        "def d(k):\n    getattr(k, 'run')(get_jot())\n"),
    "opti_oignon/s_task.py": (
        "from opti_oignon.jot import get_jot\n\n\ndef _persist(store, text):\n    store.put(text)\n\n\n"
        "def d(tasks):\n    tasks.add_task(_persist, get_jot(), 'x')\n"),
    "opti_oignon/s_thread.py": (
        "import threading\n\nfrom opti_oignon.jot import get_jot\n\n\ndef _run(store):\n    store.drop('k')\n\n\n"
        "def d():\n    threading.Thread(target=_run, args=(get_jot(),)).start()\n"),
    "opti_oignon/s_map.py": (
        "from opti_oignon.jot import get_jot\n\n\ndef _one(store):\n    store.put('x')\n\n\ndef d():\n"
        "    list(map(_one, [get_jot()]))\n"),
    "opti_oignon/s_injected.py": (
        "from opti_oignon.jot import get_jot\n\n\ndef with_store(fn):\n    def wrapper(*a):\n        return fn(get_jot(), *a)\n"
        "    return wrapper\n\n\n@with_store\ndef save(store, x):\n    store.put(x)\n"),
    "opti_oignon/s_dataclass.py": (
        "from dataclasses import dataclass\n\nfrom opti_oignon.jot import get_jot\n\n\n@dataclass\nclass Ctx:\n"
        "    store: object\n\n\ndef f():\n    Ctx(get_jot()).store.put('x')\n"),
    "opti_oignon/s_namedtuple.py": (
        "from collections import namedtuple\n\nfrom opti_oignon.jot import get_jot\n\nPair = namedtuple('Pair', ['store', 'tag'])\n"
        "\n\ndef f():\n    Pair(get_jot(), 't').store.put('x')\n"),
    "opti_oignon/s_ctx_home.py": "from dataclasses import dataclass\n\n\n@dataclass\nclass Ctx2:\n    store: object\n",
    "opti_oignon/s_dataclass_across.py": (
        "from opti_oignon.jot import get_jot\nfrom opti_oignon.s_ctx_home import Ctx2\n\n\ndef f():\n"
        "    Ctx2(store=get_jot()).store.put('x')\n"),
    "opti_oignon/s_star_home.py": "def wr(store):\n    store.put('x')\n",
    "opti_oignon/s_star.py": (
        "from opti_oignon.jot import get_jot\nfrom opti_oignon.s_star_home import *\n\n\ndef f():\n    wr(get_jot())\n"),
    # A function of another module handed with its arguments; a package that re-exports a store's symbols by a
    # star; a module relayed by a star, through a package or a plain module; a top package relayed by a plain
    # ``import``.
    "opti_oignon/s_tasks_home.py": "def persist(store, text):\n    store.put(text)\n",
    "opti_oignon/s_task_across.py": (
        "from opti_oignon.jot import get_jot\nfrom opti_oignon.s_tasks_home import persist\n\n\n"
        "def d(tasks):\n    tasks.add_task(persist, get_jot(), 'x')\n"),
    "opti_oignon/s_reexport/__init__.py": "from opti_oignon.jot import *\n",
    "opti_oignon/s_reexport_user.py": "from opti_oignon.s_reexport import get_jot\n\n\ndef f():\n    get_jot().put('x')\n",
    "opti_oignon/s_relay5.py": "from opti_oignon import jot as storage\n",
    "opti_oignon/s_pkg_star/__init__.py": "from opti_oignon.s_relay5 import *\n",
    "opti_oignon/s_star_relay.py": "from opti_oignon import s_pkg_star\n\n\ndef f():\n    s_pkg_star.storage.put_line('x')\n",
    "opti_oignon/s_plain_star.py": "from opti_oignon.s_relay5 import *\n",
    "opti_oignon/s_star_plain.py": (
        "from opti_oignon import s_plain_star\n\n\ndef f():\n    s_plain_star.storage.put_line('x')\n"),
    "opti_oignon/s_top_relay.py": "import opti_oignon.jot\n",
    "opti_oignon/s_top_user.py": (
        "from opti_oignon import s_top_relay\n\n\ndef f():\n    s_top_relay.opti_oignon.jot.put_line('x')\n"),
    # A function a package re-exports by a star, handed a store by a caller of the package; a class's own
    # factory, defined in another module.
    "opti_oignon/s_pkx/__init__.py": "from .home import *\n",
    "opti_oignon/s_pkx/home.py": "def wr2(store):\n    store.put('x')\n",
    "opti_oignon/s_pkx_user.py": (
        "from opti_oignon.jot import get_jot\nfrom opti_oignon.s_pkx import wr2\n\n\ndef f():\n    wr2(get_jot())\n"),
    "opti_oignon/s_factory_home.py": (
        "from opti_oignon.jot import Jot\n\n\nclass Made2(Jot):\n    @classmethod\n    def make(cls):\n        return cls()\n"),
    "opti_oignon/s_factory_user.py": (
        "from opti_oignon.s_factory_home import Made2\n\n\ndef f():\n    Made2.make().put('x')\n"),
}


def test_sw2_every_binding_form_is_followed_and_nothing_else_is_a_site():
    guard, restore = _load()
    try:
        _tables(guard)
        census = _census(guard, _FORMS)
    finally:
        restore()
    sites = _sites(census)
    sites.pop(_JOT, None)
    assert sites == {
        "opti_oignon/f_plain.py": [("jot", "put", "f")],
        "opti_oignon/f_alias.py": [("jot", "put", "f")],
        "opti_oignon/sub/f_relative.py": [("jot", "put", "f")],
        "opti_oignon/f_local_import.py": [("jot", "put", "f")],
        "opti_oignon/f_try.py": [("jot", "put", "f")],
        "opti_oignon/f_returner.py": [("jot", "put", "f")],
        "opti_oignon/f_method.py": [("jot", "put", "E.f")],
        "opti_oignon/f_attribute.py": [("jot", "put", "E.f")],
        "opti_oignon/f_argument.py": [("jot", "put", "w")],
        "opti_oignon/f_instance.py": [("jot", "drop", "f")],
        "opti_oignon/f_sink.py": [("jot", "put_line", "f")],
        "opti_oignon/f_handed.py": [("jot", "put", "f")],
        "opti_oignon/f_getattr.py": [("jot", "<getattr>", "f"), ("jot", "put", "f")],
        "opti_oignon/f_dotted.py": [("jot", "put", "f")],
        "opti_oignon/f_container.py": [("jot", "put", "f")],
        "opti_oignon/f_property.py": [("jot", "put", "P.f")],
        "opti_oignon/f_subclass.py": [("jot", "put", "MyJot.save")],
        "opti_oignon/f_private.py": [("jot", "_rows", "f")],
    }, sites
    # A module that imports no store is read by its syntax tree, found inert, and never walked.
    assert census.modules.get("opti_oignon/n_unrelated.py") is None or (
        census.modules["opti_oignon/n_unrelated.py"].inert and not census.records("opti_oignon/n_unrelated.py"))
    # A dependency, a context manager or a generator that yields; an element through next, enumerate, zip,
    # sorted or a literal's items; a partial provider, a class's own factory, super(), an object called; an
    # annotated dependency, a provider as a default, *args and **kwargs, a match capture, setattr, getattr
    # called; a function handed with its arguments to a task, a thread or map; an injecting decorator; a
    # dataclass and a named tuple, here and across modules; a star import's function.
    guard, restore = _load()
    try:
        _tables(guard)
        six = _sites(_census(guard, _FORMS_SIX))
    finally:
        restore()
    put = [("jot", "put", "f")]
    expected = {rel: put for rel in _FORMS_SIX}
    expected.update({
        "opti_oignon/s_super.py": [("jot", "put", "Sup.save")],
        "opti_oignon/s_varargs.py": [("jot", "drop", "w"), ("jot", "put", "w")],
        "opti_oignon/s_setattr.py": [("jot", "put", "H.f")],
        "opti_oignon/s_getattr_call.py": [("jot", "drop", "K.run")],
        "opti_oignon/s_task.py": [("jot", "put", "_persist")],
        "opti_oignon/s_thread.py": [("jot", "drop", "_run")],
        "opti_oignon/s_map.py": [("jot", "put", "_one")],
        "opti_oignon/s_injected.py": [("jot", "put", "save")],
        "opti_oignon/s_ctx_home.py": None,
        "opti_oignon/s_star_home.py": [("jot", "put", "wr")],
        "opti_oignon/s_star.py": None,
        "opti_oignon/s_tasks_home.py": [("jot", "put", "persist")],
        "opti_oignon/s_task_across.py": None,
        "opti_oignon/s_reexport/__init__.py": None,
        "opti_oignon/s_relay5.py": None,
        "opti_oignon/s_pkg_star/__init__.py": None,
        "opti_oignon/s_star_relay.py": [("jot", "put_line", "f")],
        "opti_oignon/s_plain_star.py": None,
        "opti_oignon/s_star_plain.py": [("jot", "put_line", "f")],
        "opti_oignon/s_top_relay.py": None,
        "opti_oignon/s_top_user.py": [("jot", "put_line", "f")],
        "opti_oignon/s_pkx/__init__.py": None,
        "opti_oignon/s_pkx/home.py": [("jot", "put", "wr2")],
        "opti_oignon/s_pkx_user.py": None,
        "opti_oignon/s_factory_home.py": None,
    })
    assert {rel: six.get(rel) for rel in _FORMS_SIX} == expected, six


# ---------------------------------------------------------------------------
# SW3 - SW6: classification and the ledger.
# ---------------------------------------------------------------------------
_OWING = "from opti_oignon.jot import get_jot\n\n\ndef a():\n    get_jot().put('x')\n\n\ndef b():\n    get_jot().put('y')\n"


def test_sw3_a_pair_nobody_classifies_is_a_violation_by_name():
    guard, restore = _load()
    try:
        _tables(guard)
        census = _census(guard, {"opti_oignon/a.py": _OWING})
        found = guard.find_unclassified(census)
    finally:
        restore()
    # The house's own write (put_line in jot.py) is housed, never reported.
    assert found == ["opti_oignon/a.py: 2 write site(s) of jot nobody houses, gates, exempts or owes; "
                     f"first at line {_line(_OWING, 'put(')}"], found


def test_sw4_an_owed_pair_is_tolerated_while_its_module_has_not_moved():
    guard, restore = _load()
    try:
        _tables(guard, ledger={"opti_oignon/a.py": {"seal": guard.digest(_OWING), "owes": {"jot": 2}}})
        census = _census(guard, {"opti_oignon/a.py": _OWING})
        answers = [guard.find_unclassified(census), guard.find_ledger_growth(census),
                   guard.find_broken_seals(census), guard.find_stale_entries(census)]
    finally:
        restore()
    assert answers == [[], [], [], []], answers


def test_sw5_an_owed_module_that_moved_or_grew_is_refused():
    moved = _OWING + "\n# a line\n"
    grown = _OWING + "\n\ndef c():\n    get_jot().drop('z')\n"
    guard, restore = _load()
    try:
        _tables(guard, ledger={"opti_oignon/a.py": {"seal": guard.digest(_OWING), "owes": {"jot": 2}}})
        moved_census = _census(guard, {"opti_oignon/a.py": moved})
        broken = guard.find_broken_seals(moved_census)
        growth_on_moved = guard.find_ledger_growth(moved_census)
        grown_census = _census(guard, {"opti_oignon/a.py": grown})
        growth = guard.find_ledger_growth(grown_census)
    finally:
        restore()
    assert broken == ["opti_oignon/a.py: owes, and its bytes moved since its debt was sealed: touch it, and you pay it"]
    assert growth_on_moved == [], growth_on_moved
    assert growth == ["opti_oignon/a.py: 3 write site(s) of jot, above the 2 it owes"], growth


def test_sw6_paying_the_debt_makes_the_entry_stale():
    paid = "from opti_oignon.jot import get_jot\n\n\ndef a():\n    return get_jot\n"
    half = "from opti_oignon.jot import get_jot\n\n\ndef a():\n    get_jot().put('x')\n"
    guard, restore = _load()
    try:
        _tables(guard, ledger={"opti_oignon/a.py": {"seal": guard.digest(_OWING), "owes": {"jot": 2}}})
        paid_stale = guard.find_stale_entries(_census(guard, {"opti_oignon/a.py": paid}))
        half_stale = guard.find_stale_entries(_census(guard, {"opti_oignon/a.py": half}))
        gone_stale = guard.find_stale_entries(_census(guard, {}))
    finally:
        restore()
    assert paid_stale == ["opti_oignon/a.py: owed, but it has no owed site; take it off the ledger"], paid_stale
    assert half_stale == ["opti_oignon/a.py: owes 2 write site(s) of jot, the census finds 1; lower it"], half_stale
    assert gone_stale == ["opti_oignon/a.py: owed, but it is gone; take it off the ledger"], gone_stale


# ---------------------------------------------------------------------------
# SW7, SW8
# ---------------------------------------------------------------------------
_GATE_HOME = "opti_oignon/gate.py"
_GATE_TEXT = (
    "def allow(x):\n    return bool(x)\n\n\ndef require(x):\n    if not x:\n        raise PermissionError(x)\n\n\n"
    "def pick(messages):\n    return [m for m in messages if m]\n"
)


def _gate_tables(guard, rel, gates, **more):
    kinds = {"allow": "verdict", "require": "raises", "pick": "filter"}
    defined = {name: guard.Gate(kinds[name], _GATE_HOME, name, (), "a test gate") for name in kinds}
    defined.update(more.pop("extra", {}))
    _tables(guard, gates=defined, gated={(rel, "jot"): tuple(gates)}, **more)


_SW7 = (
    "from opti_oignon.gate import allow\nfrom opti_oignon.jot import get_jot\n\n\n"
    "def guarded(x):\n    if not allow(x):\n        return None\n    get_jot().put(x)\n\n\n"
    "def bare(x):\n    get_jot().put(x)\n"
)


def test_sw7_a_gated_site_needs_its_gate_before_it():
    guard, restore = _load()
    try:
        _gate_tables(guard, "opti_oignon/a.py", ("allow",))
        found = guard.find_ungated(_census(guard, {"opti_oignon/a.py": _SW7, _GATE_HOME: _GATE_TEXT}))
    finally:
        restore()
    bare = _line(_SW7, "def bare") + 1
    assert found == [f"opti_oignon/a.py:{bare}: a jot write (put) in bare() that none of its gates (allow) covers"], found


def test_sw8_an_exemption_with_no_site_is_stale():
    quiet = "from opti_oignon.jot import get_jot\n\n\ndef a():\n    return get_jot()\n"
    guard, restore = _load()
    try:
        _tables(guard, exempt={("opti_oignon/a.py", "jot"): ("quiet", "a reason"),
                               ("opti_oignon/gone.py", "jot"): ("quiet", "a reason")})
        stale = guard.find_stale_entries(_census(guard, {"opti_oignon/a.py": quiet}))
    finally:
        restore()
    assert stale == [
        "opti_oignon/a.py: exempt for jot, but it has no jot write site; take the entry off",
        "opti_oignon/gone.py: exempt for jot, but it is gone; take the entry off",
    ], stale


# ---------------------------------------------------------------------------
# SW9 - SW12: the estate and the entry point.
# ---------------------------------------------------------------------------
def _write_tree(root, modules):
    for rel, text in modules.items():
        path = Path(root) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(text, bytes):
            path.write_bytes(text)
        else:
            path.write_text(text, encoding="utf-8")


def test_sw9_what_cannot_be_read_listed_or_parsed_fails_by_name(tmp_path):
    latin = tmp_path / "latin"
    _write_tree(latin, {_JOT: _JOT_TEXT, "opti_oignon/bad.py": b"# caf\xe9\nX = 1\n"})
    broken = tmp_path / "broken"
    _write_tree(broken, {_JOT: _JOT_TEXT, "opti_oignon/broken.py": "from opti_oignon.jot import get_jot\n\ndef f(:\n"})
    locked = tmp_path / "locked"
    _write_tree(locked, {_JOT: _JOT_TEXT, "opti_oignon/closed/m.py": "X = 1\n"})
    closed = locked / "opti_oignon" / "closed"
    guard, restore = _load()
    real_walk = guard.os.walk

    def walk(top, onerror=None, **kwargs):
        # A directory the walk cannot list, whoever runs the test, root included.
        for current, dirs, names in real_walk(top, onerror=onerror, **kwargs):
            if Path(current) == closed:
                onerror(PermissionError(13, "Permission denied", str(closed)))
                dirs[:] = []
                continue
            yield current, dirs, names

    try:
        _tables(guard)
        latin_result = guard.run(latin)
        broken_result = guard.run(broken)
        guard.os.walk = walk
        try:
            locked_result = guard.run(locked)
        finally:
            guard.os.walk = real_walk
    finally:
        restore()
    assert latin_result.code == 1 and any("opti_oignon/bad.py: cannot be read as UTF-8 text" in line
                                          for line in latin_result.lines), latin_result.lines
    assert broken_result.code == 1 and broken_result.lines[0].startswith("Write census: FAILED -- these modules do "
                                                                         "not parse"), broken_result.lines
    assert any(line.startswith("  opti_oignon/broken.py: SyntaxError") for line in broken_result.lines)
    assert locked_result.code == 1 and any("opti_oignon/closed: cannot be listed" in line
                                           for line in locked_result.lines), locked_result.lines


def _modules_on_disk():
    package = REPO / "opti_oignon"
    count = 0
    for current, dirs, names in os.walk(package):
        dirs[:] = [d for d in dirs if d not in ("data", "__pycache__")]
        count += sum(1 for name in names if name.endswith(".py"))
    return count


def test_sw10_the_real_tree_satisfies_its_own_guard_with_its_denominators():
    guard, restore = _load()
    try:
        result = _real(guard)
        answers = guard.find_all(result.census)
        tables = (len(guard.STORES), len(guard.GATED), len(guard.GATES), len(guard.EXEMPT), len(guard.LEDGER),
                  len(guard.OUTSIDE))
    finally:
        restore()
    assert result.code == 0, "\n".join(result.lines)
    assert all(found == [] for found in answers.values()), {k: v for k, v in answers.items() if v}
    line = result.lines[-1]
    stores, gated, gates, exempt, owed, outside = tables
    assert line.startswith(f"Write census OK: {_modules_on_disk()} module(s) read, "), line
    assert f" {stores} store(s);" in line and f"in {gated} pair(s) behind {gates} proven gate(s)" in line, line
    assert f"exempt in {exempt} pair(s)" in line and f"owed in {owed} module(s)" in line, line
    assert f"{outside} module(s) that open a database are outside the census by name" in line, line
    assert re.search(r"; (\d+) write site\(s\): (\d+) housed, ", line), line


def test_sw11_the_entry_point_refuses_a_violation_and_an_empty_estate(tmp_path, capsys):
    violating = tmp_path / "violating"
    _write_tree(violating, {_JOT: _JOT_TEXT, "opti_oignon/a.py": _OWING})
    empty = tmp_path / "empty"
    (empty / "opti_oignon").mkdir(parents=True)
    guard, restore = _load()
    try:
        _tables(guard)
        violating_code = guard.main(["write_census_guard.py", str(violating)])
        violating_out = capsys.readouterr().out
        empty_code = guard.main(["write_census_guard.py", str(empty)])
        empty_out = capsys.readouterr().out
    finally:
        restore()
    assert violating_code == 1, violating_out
    assert "Write census: unclassified (1):" in violating_out
    assert "opti_oignon/a.py: 2 write site(s) of jot" in violating_out, violating_out
    assert empty_code == 1 and "nothing was scanned" in empty_out, empty_out


def test_sw12_every_seal_is_a_full_digest_of_its_module():
    guard, restore = _load()
    try:
        ledger = dict(guard.LEDGER)
        seal = guard.digest("x = 1\n")
        result = _real(guard)
    finally:
        restore()
    assert seal == hashlib.sha256(b"x = 1\n").hexdigest()
    assert ledger, "the ledger holds the debt found when the guard was written"
    for rel, entry in ledger.items():
        assert re.fullmatch(r"[0-9a-f]{64}", entry["seal"]), (rel, entry["seal"])
        assert entry["seal"] == hashlib.sha256(result.estate.modules[rel].encode("utf-8")).hexdigest(), rel
        assert entry["owes"] and all(isinstance(n, int) and n > 0 for n in entry["owes"].values()), (rel, entry)


# ---------------------------------------------------------------------------
# SW13: across modules.
# ---------------------------------------------------------------------------
_ACROSS = {
    "opti_oignon/helper.py": "def write_it(store, x):\n    store.put(x)\n",
    "opti_oignon/caller.py": (
        "from opti_oignon.helper import write_it\nfrom opti_oignon.jot import get_jot\n\n\n"
        "def f():\n    write_it(get_jot(), 'x')\n"),
    "opti_oignon/state.py": "from opti_oignon.jot import Jot\n\n\nclass State:\n    def __init__(self):\n        self.j = Jot()\n",
    "opti_oignon/user.py": "def go(state):\n    state.j.drop('x')\n",
    "opti_oignon/maker.py": (
        "from opti_oignon.state import State\nfrom opti_oignon.user import go\n\n\ndef f():\n    go(State())\n"),
    "opti_oignon/svc.py": "class Svc:\n    def take(self, store):\n        store.put('x')\n",
    "opti_oignon/use.py": (
        "from opti_oignon import svc\nfrom opti_oignon.jot import get_jot\n\n\ndef f(s):\n    s.take(get_jot())\n"),
    "opti_oignon/deps.py": "from opti_oignon.jot import get_jot\n\n\ndef dep():\n    return get_jot()\n",
    "opti_oignon/route.py": (
        "from opti_oignon.deps import dep\n\n\ndef r(s=Depends(dep)):\n    s.put('x')\n"),
    "opti_oignon/hub.py": "from opti_oignon.jot import get_jot\n\nshared = get_jot()\n",
    "opti_oignon/leaf.py": "from opti_oignon.hub import shared\n\n\ndef f():\n    shared.put('x')\n",
    # A function re-exported by a package receives the store where it is defined.
    "opti_oignon/pkg/__init__.py": "from opti_oignon.pkg.impl import write_it\n",
    "opti_oignon/pkg/impl.py": "def write_it(store, x):\n    store.drop(x)\n",
    "opti_oignon/pkg_caller.py": (
        "from opti_oignon.pkg import write_it\nfrom opti_oignon.jot import get_jot\n\n\n"
        "def f():\n    write_it(get_jot(), 'x')\n"),
    # A carrier's method that hands a store object back.
    "opti_oignon/holder.py": (
        "from opti_oignon.jot import get_jot\n\n\nclass Holder:\n    def store(self):\n        return get_jot()\n"),
    "opti_oignon/use_holder.py": "from opti_oignon.holder import Holder\n\n\ndef f():\n    Holder().store().put('x')\n",
    # A subclass, in another module, of a class that keeps a store on itself.
    "opti_oignon/base.py": (
        "from opti_oignon.jot import get_jot\n\n\nclass A:\n    def __init__(self):\n        self.j = get_jot()\n"),
    "opti_oignon/sub.py": "from opti_oignon.base import A\n\n\nclass B(A):\n    def f(self):\n        self.j.put('x')\n",
    # A package's lazy export table.
    "opti_oignon/lazy/__init__.py": (
        '_EXPORTS = {"shared_jot": (".holder2", "shared_jot")}\n\n\ndef __getattr__(name):\n'
        "    raise AttributeError(name)\n"),
    "opti_oignon/lazy/holder2.py": "from opti_oignon.jot import get_jot\n\nshared_jot = get_jot()\n",
    "opti_oignon/lazy_user.py": "from opti_oignon.lazy import shared_jot\n\n\ndef f():\n    shared_jot.put('x')\n",
}


def test_sw13_a_store_object_is_followed_across_modules():
    guard, restore = _load()
    try:
        _tables(guard)
        census = _census(guard, _ACROSS)
    finally:
        restore()
    sites = _sites(census)
    sites.pop(_JOT, None)
    assert sites == {
        "opti_oignon/helper.py": [("jot", "put", "write_it")],
        "opti_oignon/user.py": [("jot", "drop", "go")],
        "opti_oignon/svc.py": [("jot", "put", "Svc.take")],
        "opti_oignon/route.py": [("jot", "put", "r")],
        "opti_oignon/leaf.py": [("jot", "put", "f")],
        "opti_oignon/pkg/impl.py": [("jot", "drop", "write_it")],
        "opti_oignon/use_holder.py": [("jot", "put", "f")],
        "opti_oignon/sub.py": [("jot", "put", "B.f")],
        "opti_oignon/lazy_user.py": [("jot", "put", "f")],
    }, sites


# ---------------------------------------------------------------------------
# SW14, SW15: gates.
# ---------------------------------------------------------------------------
_DOMINANCE = '''from contextlib import suppress

from opti_oignon.gate import allow, pick, require
from opti_oignon.jot import get_jot, put_line


def early(x):
    if not allow(x):
        return None
    get_jot().put(x)


def inside(x):
    if x and allow(x):
        get_jot().put(x)


def bound(x):
    ok = allow(x)
    if x is None or not ok:
        return None
    get_jot().put(x)


def closure(x):
    if not allow(x):
        return None

    def job():
        get_jot().put(x)
    job()


def raising(x):
    require(x)
    get_jot().put(x)


def approved(x, approve=False):
    if not approve:
        return None
    get_jot().put(x)


def filtered(messages):
    typed = pick(messages)
    put_line(typed)


def branch(x):
    if x:
        if not allow(x):
            return None
    get_jot().put(x)


def ignored(x):
    allow(x)
    get_jot().put(x)


def after(x):
    get_jot().put(x)
    if not allow(x):
        return None


def in_try(x):
    try:
        if not allow(x):
            return None
    except Exception:
        pass
    get_jot().put(x)


def rebound(x):
    ok = allow(x)
    ok = True
    if not ok:
        return None
    get_jot().put(x)


def raising_in_branch(x):
    if x:
        require(x)
    get_jot().put(x)


def unchecked(x, approve=False):
    get_jot().put(x)


def unfiltered(messages):
    put_line(messages)


def rebound_approval(x, approve=False):
    approve = True
    if not approve:
        return None
    get_jot().put(x)


def shadowed(x, ok=True):
    if x:
        ok = allow(x)
    if not ok:
        return None
    get_jot().put(x)


def swallowed(x):
    if not allow(x):
        with suppress(PermissionError):
            raise PermissionError(x)
    get_jot().put(x)


def half_filtered(messages):
    get_jot().put(pick(messages), messages)


def extended(messages):
    typed = pick(messages)
    typed.append("raw")
    put_line(typed)
'''


def test_sw14_a_gate_counts_by_dominance_never_by_line_order():
    guard, restore = _load()
    try:
        approval = {"approve": guard.Gate("approval", "opti_oignon/a.py", "approve", (), "a test gate",
                                          ("opti_oignon/a.py",))}
        _gate_tables(guard, "opti_oignon/a.py", ("allow", "require", "pick", "approve"), extra=approval)
        found = guard.find_ungated(_census(guard, {"opti_oignon/a.py": _DOMINANCE, _GATE_HOME: _GATE_TEXT}))
    finally:
        restore()
    refused = sorted(re.search(r" in (\w+)\(\)", line).group(1) for line in found)
    assert refused == sorted(["branch", "ignored", "after", "in_try", "rebound", "raising_in_branch", "unchecked",
                              "unfiltered", "rebound_approval", "shadowed", "swallowed", "half_filtered",
                              "extended"]), found


_DECIDED = '''from opti_oignon.jot import get_jot


def apply(x):
    get_jot().put(x)


class Gatekeeper:
    def write(self, x):
        if x:
            return apply(x)
        return None


def _private(x):
    get_jot().drop(x)


def accept(x):
    return _private(x)
'''


def test_sw15_a_function_is_gated_by_its_callers_only_when_every_caller_is():
    elsewhere = "from opti_oignon.a import apply\n\n\ndef f():\n    apply('y')\n"
    handed = _DECIDED + "\n\ndef hand():\n    return _private\n"
    # References the rule must see, each from outside the gates.
    starred = "from opti_oignon.a import *\n\n\ndef f():\n    apply('y')\n"
    renamed = {"opti_oignon/pkg/__init__.py": "from opti_oignon.a import apply as run_it\n",
               "opti_oignon/c.py": "from opti_oignon.pkg import run_it\n\n\ndef f():\n    run_it('z')\n"}
    decorated = _DECIDED.replace("\n\ndef apply(x):", "\n\n@register\ndef apply(x):")
    aliased = _DECIDED + "\n\nclass K:\n    alias = _private\n"
    fetched = "import opti_oignon.a as a_mod\n\n\ndef f():\n    return getattr(a_mod, 'apply')\n"
    guard, restore = _load()
    try:
        gates = {"keeper": guard.Gate("decision", "opti_oignon/a.py", "Gatekeeper.write", (), "a test gate"),
                 "accept": guard.Gate("decision", "opti_oignon/a.py", "accept", (), "a test gate")}
        _tables(guard, gates=gates, gated={("opti_oignon/a.py", "jot"): ("keeper", "accept")})

        def refused(modules):
            found = guard.find_ungated(_census(guard, modules))
            return sorted(re.search(r" in (\w+)\(\)", line).group(1) for line in found)

        alone = refused({"opti_oignon/a.py": _DECIDED})
        reached = refused({"opti_oignon/a.py": _DECIDED, "opti_oignon/b.py": elsewhere})
        leaked = refused({"opti_oignon/a.py": handed})
        star = refused({"opti_oignon/a.py": _DECIDED, "opti_oignon/b.py": starred})
        rename = refused({"opti_oignon/a.py": _DECIDED, **renamed})
        decorator = refused({"opti_oignon/a.py": decorated})
        alias = refused({"opti_oignon/a.py": aliased})
        getter = refused({"opti_oignon/a.py": _DECIDED, "opti_oignon/d.py": fetched})
    finally:
        restore()
    assert alone == [], alone
    assert reached == ["apply"], reached
    assert leaked == ["_private"], leaked
    assert star == ["apply"], star
    assert rename == ["apply"], rename
    assert decorator == ["apply"], decorator
    assert alias == ["_private"], alias
    assert getter == ["apply"], getter


# ---------------------------------------------------------------------------
# SW16: checked exemptions.
# ---------------------------------------------------------------------------
_ROUTES = '''from fastapi import APIRouter

from opti_oignon.jot import get_jot

router = APIRouter()


@router.post("/x")
def handler(x):
    _helper(x)
    get_jot().put(x)


def _helper(x):
    get_jot().put(x)


def elsewhere(x):
    get_jot().put(x)
'''


def test_sw16_each_checked_exemption_is_refused_where_its_predicate_fails():
    keyed = ("from opti_oignon.jot import get_jot\n\n\ndef f(c):\n    get_jot().put('a', context=c)\n"
             "    get_jot().put('b')\n    get_jot().put('c', context=None)\n")
    quiet = "from opti_oignon.jot import get_jot\n\n\ndef f():\n    get_jot().drop('a')\n    get_jot().put('b')\n"
    script = ("from opti_oignon.jot import get_jot\n\n\ndef f():\n    get_jot().put('a')\n\n\n"
              "if __name__ == '__main__':\n    get_jot().put('b')\n")
    program = ("from opti_oignon.jot import get_jot\n\nget_jot().put('a')\n\n\n"
               "def run():\n    get_jot().put('b')\n")
    caller = "from opti_oignon.tool.__main__ import run\n\n\ndef go():\n    run()\n"
    own = ("import tempfile\nfrom pathlib import Path\n\nfrom opti_oignon.jot import Jot\n\n\n"
           "def f():\n    with tempfile.TemporaryDirectory() as tmp:\n        j = Jot(path=Path(tmp) / 'j.db')\n"
           "        j.put('a')\n")
    borrowed = "from opti_oignon.jot import get_jot\n\n\ndef f():\n    get_jot().put('a')\n"
    unplaced = "from opti_oignon.jot import Jot\n\n\ndef f():\n    Jot().put('a')\n"
    nowhere = "from opti_oignon.jot import Jot\n\n\ndef f():\n    Jot(None).put('a')\n"
    homely = "from opti_oignon.jot import Jot\n\nUSER_DB = '/home/user/jot.db'\n\n\ndef f():\n    Jot(path=USER_DB).put('a')\n"
    sinking = ("import tempfile\n\nfrom opti_oignon.jot import Jot, put_line\n\n\n"
               "def f():\n    Jot(path=tempfile.mkdtemp()).put('a')\n    put_line('b')\n")
    writer = "from opti_oignon.jot import get_jot\n\n\ndef f():\n    get_jot().put('a')\n"
    reader = "from opti_oignon.jot import get_jot\n\n\ndef g():\n    return get_jot().read_back()\n"
    modules = {"opti_oignon/routes.py": _ROUTES, "opti_oignon/keyed.py": keyed, "opti_oignon/quiet.py": quiet,
               "opti_oignon/script.py": script, "opti_oignon/tool/__init__.py": "",
               "opti_oignon/tool/__main__.py": program, "opti_oignon/own.py": own, "opti_oignon/borrowed.py": borrowed,
               "opti_oignon/unplaced.py": unplaced, "opti_oignon/nowhere.py": nowhere, "opti_oignon/homely.py": homely,
               "opti_oignon/sinking.py": sinking, "opti_oignon/writer.py": writer}
    exempt = {("opti_oignon/routes.py", "jot"): ("route", "r"), ("opti_oignon/keyed.py", "jot"): ("keyed", "r"),
              ("opti_oignon/quiet.py", "jot"): ("quiet", "r"), ("opti_oignon/script.py", "jot"): ("script", "r"),
              ("opti_oignon/tool/__main__.py", "jot"): ("script", "r"), ("opti_oignon/own.py", "jot"): ("instance", "r"),
              ("opti_oignon/borrowed.py", "jot"): ("instance", "r"), ("opti_oignon/unplaced.py", "jot"): ("instance", "r"),
              ("opti_oignon/nowhere.py", "jot"): ("instance", "r"), ("opti_oignon/homely.py", "jot"): ("instance", "r"),
              ("opti_oignon/sinking.py", "jot"): ("instance", "r"), ("opti_oignon/writer.py", "jot"): ("unread", "r")}
    guard, restore = _load()
    try:
        _tables(guard, exempt=exempt)
        unread_alone = guard.find_failed_exemptions(_census(guard, modules))
        with_reader = guard.find_failed_exemptions(_census(guard, {**modules, "opti_oignon/reader.py": reader}))
        called = guard.find_failed_exemptions(_census(guard, {**modules, "opti_oignon/caller.py": caller}))
    finally:
        restore()
    elsewhere = _line(_ROUTES, "def elsewhere") + 1
    keyed_lines = [_line(keyed, "put('b')"), _line(keyed, "put('c'")]
    foreign = "a jot write on an object the module did not build itself at a temporary place"
    assert sorted(unread_alone) == sorted([
        f"opti_oignon/routes.py:{elsewhere}: a jot write in elsewhere() that no route handler alone reaches",
        f"opti_oignon/keyed.py:{keyed_lines[0]}: a jot write that binds no context key (context)",
        f"opti_oignon/keyed.py:{keyed_lines[1]}: a jot write that binds no context key (context)",
        f"opti_oignon/quiet.py:{_line(quiet, 'put(')}: put adds content to jot",
        f"opti_oignon/script.py:{_line(script, 'put(' )}: a jot write outside the module's __main__ block",
        f"opti_oignon/borrowed.py:{_line(borrowed, 'put(')}: {foreign}",
        f"opti_oignon/unplaced.py:{_line(unplaced, 'put(')}: {foreign}",
        f"opti_oignon/nowhere.py:{_line(nowhere, 'put(')}: {foreign}",
        f"opti_oignon/homely.py:{_line(homely, 'put(')}: {foreign}",
        f"opti_oignon/sinking.py:{_line(sinking, 'put_line(')}: {foreign}",
    ]), unread_alone
    # A program module another module imports is a program no more: the import runs its module-level write,
    # and the caller calls its function.
    assert sorted(set(called) - set(unread_alone)) == sorted([
        f"opti_oignon/tool/__main__.py:{_line(program, 'put(')}: a jot write outside the module's __main__ block",
        f"opti_oignon/tool/__main__.py:{_line(program, 'def run') + 1}: a jot write outside the module's __main__ block",
    ]), called
    assert sorted(set(with_reader) - set(unread_alone)) == [
        f"opti_oignon/reader.py:{_line(reader, 'read_back')}: reads jot back (read_back), so its writes are read"
    ], with_reader
    # A read reached by ``getattr`` with its name reads the store back too.
    by_name = "from opti_oignon.jot import get_jot\n\n\ndef g():\n    return getattr(get_jot(), 'read_back')()\n"
    guard, restore = _load()
    try:
        _tables(guard, exempt=exempt)
        by_getattr = guard.find_failed_exemptions(_census(guard, {**modules, "opti_oignon/reader2.py": by_name}))
    finally:
        restore()
    assert sorted(set(by_getattr) - set(unread_alone)) == [
        f"opti_oignon/reader2.py:{_line(by_name, 'read_back')}: reads jot back (read_back), so its writes are read"
    ], by_getattr


# ---------------------------------------------------------------------------
# SW17 - SW20: the tables.
# ---------------------------------------------------------------------------
def test_sw17_a_gate_is_defined_at_home_and_proven_by_a_suite_that_names_it(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_g_contracts.py").write_text(
        "import opti_oignon.gate\n\n\ndef test_g1_allow_refuses():\n    pass\n", encoding="utf-8")
    (tmp_path / "tests" / "test_h_contracts.py").write_text(
        "_MODULES = (\"gate\",)\n\n\ndef test_h1_quoted_stem():\n    pass\n", encoding="utf-8")
    (tmp_path / "tests" / "test_k_contracts.py").write_text(
        "def test_k1_names_nothing():\n    pass\n", encoding="utf-8")
    guard, restore = _load()
    try:
        gates = {
            "proven": guard.Gate("verdict", _GATE_HOME, "allow", ("g1", "h1"), "r"),
            "unproven": guard.Gate("verdict", _GATE_HOME, "allow", ("g9",), "r"),
            "elsewhere": guard.Gate("verdict", _GATE_HOME, "allow", ("k1",), "r"),
            "missing": guard.Gate("raises", _GATE_HOME, "nothing_here", ("g1",), "r"),
            "odd": guard.Gate("prayer", _GATE_HOME, "allow", ("g1",), "r"),
        }
        _tables(guard, gates=gates, gated={("opti_oignon/a.py", "jot"): ("proven", "absent")},
                exempt={("opti_oignon/b.py", "jot"): ("because", "r"), ("opti_oignon/c.py", "jot"): ("quiet", " ")})
        census = _census(guard, {_GATE_HOME: _GATE_TEXT}, root=tmp_path)
        proofs = guard.find_gate_proofs_missing(census)
        unknown = guard.find_unknown_gates(census)
    finally:
        restore()
    assert sorted(proofs) == sorted([
        "gate unproven: contract g9 names no test function that runs",
        "gate elsewhere: the suite of contract k1 (tests/test_k_contracts.py) never names opti_oignon/gate.py",
    ]), proofs
    assert sorted(unknown) == sorted([
        "opti_oignon/a.py: jot names the gate absent, which GATES does not define",
        "gate missing: opti_oignon/gate.py does not define nothing_here",
        "gate odd: kind 'prayer' is none of acceptance, approval, decision, filter, house, raises, verdict",
        "opti_oignon/b.py: jot is exempt as 'because', a kind no predicate checks and no reason argues",
        "opti_oignon/c.py: jot is exempt with no reason",
    ]), unknown


def test_sw18_a_store_the_table_misdescribes_is_refused():
    housed = _JOT_TEXT.replace("    def put(self, text, context=None):\n",
                               "    def put(self, text, context=None):\n        self._check(text)\n")
    guard, restore = _load()
    try:
        wrong = _jot(guard, classes=("Jot", "Ghost"), accesses=("get_jot", "get_ghost"), instances=("jot", "phantom"),
                     writes=("put", "drop", "erase"), quiet=("drop", "shred"), backs=("nowhere",))
        _tables(guard, stores={"jot": wrong})
        drift = guard.find_table_drift(_census(guard, {}))
        _tables(guard, gates={"own": guard.Gate("house", _JOT, "_check", (), "r")})
        house = guard.find_table_drift(_census(guard, {}))
        guarded = housed.replace("    def drop(self, key):\n", "    def drop(self, key):\n        self._check(key)\n")
        house_ok = guard.find_table_drift(guard.take_census(guard.Estate(
            Path("/nonexistent-write-census"), {_JOT: guarded}, [])))
        # A house gate given a constant says yes to everyone.
        constant = guarded.replace("self._check(text)", "self._check('user')")
        house_constant = guard.find_table_drift(guard.take_census(guard.Estate(
            Path("/nonexistent-write-census"), {_JOT: constant}, [])))
        # The methods of a store are closed: a method that writes is listed or set aside.
        extra = _JOT_TEXT.replace(
            "    def read_back(self):\n",
            "    def purge(self):\n        self.drop('all')\n\n    def reset(self):\n"
            "        return 'DELETE FROM rows'\n\n    def read_back(self):\n")
        _tables(guard)
        unlisted = guard.find_table_drift(guard.take_census(guard.Estate(
            Path("/nonexistent-write-census"), {_JOT: extra}, [])))
        _tables(guard, stores={"jot": _jot(guard, aside={"purge": "r", "reset": "r"})})
        set_aside = guard.find_table_drift(guard.take_census(guard.Estate(
            Path("/nonexistent-write-census"), {_JOT: extra}, [])))
        _tables(guard, stores={"jot": _jot(guard, aside={"purge": " ", "reset": "r", "put": "r"})})
        misaside = guard.find_table_drift(guard.take_census(guard.Estate(
            Path("/nonexistent-write-census"), {_JOT: extra}, [])))
    finally:
        restore()
    assert house_constant == ["gate own: Jot.put writes before it calls _check"], house_constant
    assert sorted(unlisted) == [
        "store jot: Jot.purge calls the write drop, and is neither a write nor set aside with a reason",
        "store jot: Jot.reset runs an SQL write, and is neither a write nor set aside with a reason",
    ], unlisted
    assert set_aside == [], set_aside
    assert sorted(misaside) == ["store jot: purge is set aside with no reason",
                                "store jot: put is both a write and set aside"], misaside
    assert sorted(drift) == sorted([
        "store jot: opti_oignon/jot.py defines no class Ghost",
        "store jot: opti_oignon/jot.py defines no function get_ghost",
        "store jot: opti_oignon/jot.py binds no phantom at module level",
        "store jot: none of Jot, Ghost defines erase",
        "store jot: shred is quiet but not a write",
        "store jot: it is built on nowhere, which STORES does not name",
    ]), drift
    assert sorted(house) == ["gate own: Jot.drop writes before it calls _check",
                             "gate own: Jot.put writes before it calls _check"], house
    assert house_ok == [], house_ok
    # An SQL write after a comment, in a script, in a common table expression, in a constant of the module or of
    # the class; a write of its class reached by name or by getattr; a private helper that writes. A docstring
    # that mentions one writes nothing.
    hidden = _JOT_TEXT.replace(
        "    def read_back(self):\n        return []\n",
        "    def a(self):\n        return '-- note\\nINSERT INTO rows VALUES (1)'\n\n"
        "    def b(self):\n        return 'BEGIN; DELETE FROM rows; COMMIT'\n\n"
        "    def c(self):\n        return 'WITH t AS (SELECT 1) UPDATE rows SET x = 1'\n\n"
        "    def d(self):\n        return _SQL\n\n"
        "    def e(self):\n        return self._SQL_CLASS\n\n"
        "    def f(self):\n        writer = self.put\n        return writer\n\n"
        "    def g(self):\n        return getattr(self, 'drop')('k')\n\n"
        "    def h(self):\n        return self._insert()\n\n"
        "    def _insert(self):\n        return 'INSERT INTO rows VALUES (2)'\n\n"
        "    def read_back(self):\n        \"\"\"Never an INSERT INTO rows, in prose.\"\"\"\n        return []\n").replace(
        "class Jot:\n", "_SQL = 'REPLACE INTO rows VALUES (3)'\n\n\nclass Jot:\n    _SQL_CLASS = 'UPDATE rows SET y = 2'\n\n", 1)
    # The house's own class carries nothing that may rebuild it or reroute its methods, and rebinds its gate
    # nowhere, its own house included.
    decorated = guarded.replace("class Jot:", "def keep(cls):\n    return cls\n\n\n@keep\nclass Jot:", 1)
    hooked = guarded.replace("    def read_back(self):\n", "    def __getattribute__(self, name):\n"
                             "        return object.__getattribute__(self, name)\n\n    def read_back(self):\n", 1)
    rebinding = guarded.replace("    def read_back(self):\n", "    def loosen(self):\n"
                                "        Jot._check = lambda s, v: None\n\n    def read_back(self):\n", 1)
    guard, restore = _load()
    try:
        _tables(guard)
        shown = guard.find_table_drift(guard.take_census(guard.Estate(
            Path("/nonexistent-write-census"), {_JOT: hidden}, [])))
        _tables(guard, gates={"own": guard.Gate("house", _JOT, "_check", (), "r")})
        own_class = {key: guard.find_table_drift(guard.take_census(guard.Estate(
            Path("/nonexistent-write-census"), {_JOT: text}, []))) for key, text in (("decorated", decorated),
                                                                                       ("hooked", hooked))}
        rebound = guard.find_unclassified(guard.take_census(guard.Estate(
            Path("/nonexistent-write-census"), {_JOT: rebinding}, [])))
    finally:
        restore()
    unlisted_tail = "and is neither a write nor set aside with a reason"
    assert sorted(shown) == sorted(
        [f"store jot: Jot.{m} runs an SQL write, {unlisted_tail}" for m in "abcde"]
        + [f"store jot: Jot.f calls the write put, {unlisted_tail}",
           f"store jot: Jot.g calls the write drop, {unlisted_tail}",
           f"store jot: Jot.h calls _insert, which writes, {unlisted_tail}"]), shown
    assert own_class == {
        "decorated": ["gate own: Jot carries a class decorator or a metaclass, which may rebuild it"],
        "hooked": ["gate own: Jot defines __getattribute__, which may reroute any of its methods"]}, own_class
    assert rebound == [f"{_JOT}: 1 write site(s) of jot nobody houses, gates, exempts or owes; first at line "
                       f"{_line(rebinding, 'Jot._check =')}"], rebound


def test_sw19_a_pair_classified_twice_is_refused():
    guard, restore = _load()
    try:
        _tables(guard, gates={"g": guard.Gate("verdict", _GATE_HOME, "allow", (), "r")},
                gated={("opti_oignon/a.py", "jot"): ("g",)}, exempt={("opti_oignon/a.py", "jot"): ("quiet", "r")},
                ledger={"opti_oignon/a.py": {"seal": "0" * 64, "owes": {"jot": 2}}})
        found = guard.find_conflicts(_census(guard, {"opti_oignon/a.py": _OWING}))
    finally:
        restore()
    assert found == ["opti_oignon/a.py: jot is classified in GATED and EXEMPT and LEDGER"], found


def test_sw20_every_module_that_opens_a_database_is_classified():
    modules = {
        "opti_oignon/db_utils.py": "import sqlite3\n",
        "opti_oignon/keeper.py": "import sqlite3\n",
        "opti_oignon/vectors.py": "from chromadb import PersistentClient\n",
        "opti_oignon/helped.py": "from .db_utils import connect\n",
        "opti_oignon/listed.py": "from opti_oignon import db_utils\n",
        "opti_oignon/named.py": "import sqlite3\n",
        "opti_oignon/talker.py": "# sqlite3 is mentioned in a comment only\nX = 1\n",
        "opti_oignon/aio.py": "import aiosqlite\n",
        "opti_oignon/dotted.py": "import opti_oignon.db_utils as du\n",
    }
    house = _JOT_TEXT.replace('"""A test store."""\n', '"""A test store."""\nimport sqlite3\n')
    guard, restore = _load()
    try:
        _tables(guard, outside={"opti_oignon/named.py": "a reason", "opti_oignon/talker.py": "a reason"},
                helpers={"opti_oignon/db_utils.py": "the helper"})
        census = guard.take_census(guard.Estate(Path("/nonexistent-write-census"),
                                                {"opti_oignon/__init__.py": "", _JOT: house, **modules}, []))
        unclassified = guard.find_unclassified_stores(census)
        stale = [line for line in guard.find_stale_entries(census) if "outside" in line]
    finally:
        restore()
    assert unclassified == [
        f"opti_oignon/{name}.py: opens a database, and is no store's house, no helper, and not named in OUTSIDE"
        for name in ("aio", "dotted", "helped", "keeper", "listed", "vectors")], unclassified
    assert stale == ["opti_oignon/talker.py: outside the census, but it opens no database; take it off"], stale
    # A database library imported by its name at run time opens one; a helper the table names opens what its
    # importers name, whatever its stem.
    dynamic = {"opti_oignon/late.py": "import importlib\n\n\ndef f():\n    return importlib.import_module('sqlite3')\n",
               "opti_oignon/later.py": "def f():\n    return __import__('sqlite3')\n",
               "opti_oignon/pool2.py": "import sqlite3\n",
               "opti_oignon/borrower.py": "from opti_oignon import pool2\n"}
    guard, restore = _load()
    try:
        _tables(guard, helpers={"opti_oignon/pool2.py": "a helper of another stem"})
        late = guard.find_unclassified_stores(guard.take_census(guard.Estate(
            Path("/nonexistent-write-census"), {"opti_oignon/__init__.py": "", _JOT: _JOT_TEXT, **dynamic}, [])))
    finally:
        restore()
    assert late == [
        f"opti_oignon/{name}.py: opens a database, and is no store's house, no helper, and not named in OUTSIDE"
        for name in ("borrower", "late", "later")], late


# ---------------------------------------------------------------------------
# SW21: two censuses agree.
# ---------------------------------------------------------------------------
_DISTINCT = (
    "add_note", "update_note", "delete_note", "add_attachment", "update_attachment", "delete_attachment",
    "apply_synced_note", "apply_synced_skill", "apply_synced_memory_canonical", "apply_synced_conversation",
    "add_message", "delete_last_message", "delete_conversation", "create_conversation", "update_conversation_metadata",
    "add_branch_message", "update_branch", "merge_messages", "delete_branch", "delete_all_branches",
    "extract_and_store", "aextract_and_store", "schedule_extraction", "save_working_memory", "store_embedding",
    "put_with_embedding", "soft_delete", "hard_delete", "supersede", "set_runtime_override", "create_project",
    "update_project", "reindex_project", "ingest_file", "ingest_url", "store_chunked", "index_directory",
    "add_fact", "update_fact", "delete_fact", "deactivate_fact", "activate_fact", "link_conversation",
    "unlink_conversation", "delete_document", "delete_collection", "remove_file_from_index", "delete_project_index",
    "clear_runtime_override", "clear_all_runtime_overrides", "ingest_text", "adopt_synced", "record_checkpoint",
    "evict_span", "evict_oldest", "take_back",
)

# Calls of a distinctive write name that write into no census store, each with
# the reason: (module, the call as written).
_NOT_CENSUS = {
    ("opti_oignon/api/routes_memory.py", "librarian.supersede"):
        "the librarian's own function; its Core write is a site in the librarian",
    ("opti_oignon/project_context.py", "client.delete_collection"):
        "the project index's own chromadb client, inside its house (ProjectIndexer.delete_project_index)",
    ("opti_oignon/plugins/scratchpad/entry_point.py", "db.add_note"): "the scratchpad plugin's own database",
    ("opti_oignon/plugins/scratchpad/entry_point.py", "db.delete_note"): "the scratchpad plugin's own database",
}


def test_sw21_a_textual_census_of_write_names_agrees_with_the_binding_census():
    guard, restore = _load()
    try:
        result = _real(guard)
        census = result.census
        lines = {(rel, s.line) for rel, found in census.sites().items() for s in found}
        houses = {spec.house for spec in guard.STORES.values()}
    finally:
        restore()
    unexplained, explained = [], set()
    for rel, text in sorted(result.estate.modules.items()):
        if not any(name in text for name in _DISTINCT):
            continue
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            # A write reached by name: getattr(x, "add_message").
            if name == "getattr" and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                name = node.args[1].value
                func = node
            if name not in _DISTINCT or (rel, node.lineno) in lines:
                continue
            written = ast.unparse(func)
            if rel in houses and written.startswith("self."):
                continue
            if (rel, written) in _NOT_CENSUS:
                explained.add((rel, written))
                continue
            unexplained.append(f"{rel}:{node.lineno} {written}")
    assert unexplained == [], unexplained
    # Each named exception is still there to be explained.
    assert explained == set(_NOT_CENSUS), sorted(set(_NOT_CENSUS) - explained)


# ---------------------------------------------------------------------------
# SW22 - SW24
# ---------------------------------------------------------------------------
_LOOKUPS = {
    "opti_oignon/look_sys.py": (
        "import sys\n\n\ndef f():\n    mod = sys.modules.get('opti_oignon.jot')\n"
        "    store = getattr(mod, 'jot', None)\n    apply = getattr(store, 'put', None)\n    apply('x')\n"),
    "opti_oignon/look_index.py": "import sys\n\n\ndef f():\n    sys.modules['opti_oignon.jot'].jot.drop('k')\n",
    "opti_oignon/look_import.py": (
        "import importlib\n\n\ndef f():\n    m = importlib.import_module('opti_oignon.jot')\n    m.get_jot().put('x')\n"),
    "opti_oignon/dispatch.py": (
        "from opti_oignon.jot import get_jot\n\n\ndef _add(store, x):\n    store.put(x)\n\n\n"
        "def _remove(store, x):\n    store.drop(x)\n\n\nWRITES = {'add': _add, 'remove': _remove}\n\n\n"
        "def handle(action, x):\n    return WRITES[action](get_jot(), x)\n"),
    "opti_oignon/dispatch_get.py": (
        "from opti_oignon.jot import get_jot\n\n\ndef _keep(store, x):\n    store.put(x)\n\n\n"
        "TABLE = {'keep': _keep}\n\n\ndef handle(action, x):\n    return TABLE.get(action)(get_jot(), x)\n"),
}


def test_sw22_a_module_looked_up_by_name_and_a_dispatch_table_are_followed():
    guard, restore = _load()
    try:
        _tables(guard)
        census = _census(guard, _LOOKUPS)
    finally:
        restore()
    sites = _sites(census)
    sites.pop(_JOT, None)
    assert sites == {
        "opti_oignon/look_sys.py": [("jot", "put", "f")],
        "opti_oignon/look_index.py": [("jot", "drop", "f")],
        "opti_oignon/look_import.py": [("jot", "put", "f")],
        "opti_oignon/dispatch.py": [("jot", "drop", "_remove"), ("jot", "put", "_add")],
        "opti_oignon/dispatch_get.py": [("jot", "put", "_keep")],
    }, sites


_APPROVAL_HOME = "opti_oignon/approve.py"
_APPROVAL = (
    "from opti_oignon.jot import get_jot\n\n\ndef caption(x, approve=False):\n    if not approve:\n"
    "        return None\n    get_jot().put(x)\n")
_QUEUE = (
    "from opti_oignon.jot import get_jot\n\n\ndef _apply(x):\n    get_jot().put(x)\n\n\n"
    "def accept(ids):\n    for x in ids:\n        _apply(x)\n")


def test_sw23_a_gate_whose_authority_is_the_callers_is_reached_only_from_the_users_gesture():
    web = ("from fastapi import APIRouter\n\nfrom opti_oignon.approve import caption\nfrom opti_oignon.queue import accept\n\n"
           "router = APIRouter()\n\n\n"
           "@router.post('/caption')\ndef caption_route(x, approve: bool = False):\n    return caption(x, approve=approve)\n\n\n"
           "@router.post('/accept')\ndef accept_route(ids):\n    return accept(ids)\n")
    model = ("from opti_oignon.approve import caption\nfrom opti_oignon.queue import accept\n\n\n"
             "def tool(x):\n    caption(x, approve=True)\n    accept([x])\n")
    guard, restore = _load()
    try:
        def tables(entries=()):
            gates = {"approved": guard.Gate("approval", _APPROVAL_HOME, "approve", (), "r", entries),
                     "accepted": guard.Gate("acceptance", "opti_oignon/queue.py", "accept", (), "r", entries)}
            _tables(guard, gates=gates, gated={(_APPROVAL_HOME, "jot"): ("approved",),
                                               ("opti_oignon/queue.py", "jot"): ("accepted",)})

        base = {_APPROVAL_HOME: _APPROVAL, "opti_oignon/queue.py": _QUEUE, "opti_oignon/web.py": web}
        tables()
        from_routes = guard.find_ungated(_census(guard, base))
        from_model = guard.find_ungated(_census(guard, {**base, "opti_oignon/model.py": model}))
        tables(entries=("opti_oignon/model.py",))
        from_entry = guard.find_ungated(_census(guard, {**base, "opti_oignon/model.py": model}))
    finally:
        restore()
    assert from_routes == [], from_routes
    assert sorted(from_model) == sorted([
        f"{_APPROVAL_HOME}:{_line(_APPROVAL, 'put(')}: a jot write whose approval gate (approved) takes its "
        f"authority from a caller that no route or named entry alone reaches",
        "opti_oignon/queue.py: accept carries the user's acceptance, and something other than a route or a "
        "named entry reaches it",
    ]), from_model
    assert from_entry == [], from_entry


def test_sw24_a_private_member_of_a_store_reached_outside_its_house_is_a_site():
    peek = ("from opti_oignon.jot import get_jot\n\n\ndef f():\n    conn = get_jot()._conn\n"
            "    kind = get_jot().__class__\n    return conn, kind\n")
    house = _JOT_TEXT.replace("    def read_back(self):\n        return []\n",
                              "    def read_back(self):\n        return self._rows\n")
    guard, restore = _load()
    try:
        _tables(guard)
        census = guard.take_census(guard.Estate(
            Path("/nonexistent-write-census"),
            {"opti_oignon/__init__.py": "", _JOT: house, "opti_oignon/peek.py": peek}, []))
        unclassified = guard.find_unclassified(census)
    finally:
        restore()
    sites = _sites(census)
    assert sites.get("opti_oignon/peek.py") == [("jot", "_conn", "f")], sites
    assert all(method != "_rows" for _store, method, _f in sites.get(_JOT, [])), sites.get(_JOT)
    assert unclassified == [f"opti_oignon/peek.py: 1 write site(s) of jot nobody houses, gates, exempts or owes; "
                            f"first at line {_line(peek, '_conn')}"], unclassified


# ---------------------------------------------------------------------------
# SW25 - SW30: what the second review found.
# ---------------------------------------------------------------------------
_FORMS_TWO = {
    "opti_oignon/l_fromlist.py": "def f():\n    m = __import__('opti_oignon.jot', fromlist=['jot'])\n    m.jot.put('x')\n",
    "opti_oignon/l_relative.py": (
        "import importlib\n\n\ndef f():\n    m = importlib.import_module('.jot', 'opti_oignon')\n    m.jot.drop('k')\n"),
    "opti_oignon/l_or.py": (
        "import importlib\nimport sys\n\n\ndef f():\n"
        "    m = sys.modules.get('opti_oignon.jot') or importlib.import_module('opti_oignon.jot')\n    m.jot.put('x')\n"),
    "opti_oignon/l_attr.py": (
        "import sys\n\n\nclass H:\n    def __init__(self):\n        self.mod = sys.modules.get('opti_oignon.jot')\n\n"
        "    def f(self):\n        self.mod.jot.put('x')\n"),
    "opti_oignon/l_modules.py": "from sys import modules\n\n\ndef f():\n    modules['opti_oignon.jot'].jot.put('x')\n",
    "opti_oignon/helper_a.py": "def write_it(store, x):\n    store.put(x)\n",
    "opti_oignon/helper_b.py": "def write_it(store, x):\n    return x\n",
    "opti_oignon/l_two.py": (
        "import importlib\n\n\ndef b():\n    m = importlib.import_module('opti_oignon.helper_b')\n\n\n"
        "def a():\n    m = importlib.import_module('opti_oignon.helper_a')\n"
        "    m.write_it(importlib.import_module('opti_oignon.jot').get_jot(), 'x')\n"),
    "opti_oignon/l_try.py": (
        "try:\n    from opti_oignon import helper_b as impl\nexcept ImportError:\n    from opti_oignon import jot as impl\n\n\n"
        "def f():\n    impl.get_jot().put('x')\n"),
    # The same two bindings in the other order: whichever comes first, both hold.
    "opti_oignon/helper_c.py": "def write_it(store, x):\n    store.drop(x)\n",
    "opti_oignon/helper_d.py": "def write_it(store, x):\n    return x\n",
    "opti_oignon/l_two_rev.py": (
        "import importlib\n\n\ndef a():\n    m = importlib.import_module('opti_oignon.helper_c')\n"
        "    m.write_it(importlib.import_module('opti_oignon.jot').get_jot(), 'x')\n\n\n"
        "def b():\n    m = importlib.import_module('opti_oignon.helper_d')\n"),
    "opti_oignon/l_try_rev.py": (
        "try:\n    from opti_oignon import jot as impl\nexcept ImportError:\n    from opti_oignon import helper_b as impl\n\n\n"
        "def f():\n    impl.get_jot().put('x')\n"),
    "opti_oignon/t_indirect.py": (
        "from opti_oignon.jot import get_jot\n\n\ndef _keep(store, x):\n    store.put(x)\n\n\nTABLE = {'keep': _keep}\n\n\n"
        "def handle(a, x):\n    fn = TABLE.get(a)\n    return fn(get_jot(), x)\n"),
    "opti_oignon/t_attr.py": (
        "from opti_oignon.jot import get_jot\n\n\ndef _drop(store, x):\n    store.drop(x)\n\n\nclass D:\n"
        "    def __init__(self):\n        self.TABLE = {'drop': _drop}\n\n    def handle(self, a, x):\n"
        "        return self.TABLE[a](get_jot(), x)\n"),
    "opti_oignon/t_lambda.py": (
        "from opti_oignon.jot import get_jot\n\nTABLE = {'put': lambda store, x: store.put(x)}\n\n\n"
        "def handle(a, x):\n    return TABLE[a](get_jot(), x)\n"),
    "opti_oignon/t_tried.py": (
        "try:\n    def write_tried(store, x):\n        store.put(x)\nexcept Exception:\n    write_tried = None\n"),
    "opti_oignon/t_tried_caller.py": (
        "from opti_oignon.jot import get_jot\nfrom opti_oignon.t_tried import write_tried\n\n\n"
        "def f():\n    write_tried(get_jot(), 'x')\n"),
}


def test_sw25_every_form_of_a_module_lookup_and_a_dispatch_table_is_followed():
    guard, restore = _load()
    try:
        _tables(guard)
        census = _census(guard, _FORMS_TWO)
    finally:
        restore()
    sites = _sites(census)
    sites.pop(_JOT, None)
    assert sites == {
        "opti_oignon/l_fromlist.py": [("jot", "put", "f")],
        "opti_oignon/l_relative.py": [("jot", "drop", "f")],
        "opti_oignon/l_or.py": [("jot", "put", "f")],
        "opti_oignon/l_attr.py": [("jot", "put", "H.f")],
        "opti_oignon/l_modules.py": [("jot", "put", "f")],
        "opti_oignon/helper_a.py": [("jot", "put", "write_it")],
        "opti_oignon/l_try.py": [("jot", "put", "f")],
        "opti_oignon/helper_c.py": [("jot", "drop", "write_it")],
        "opti_oignon/l_try_rev.py": [("jot", "put", "f")],
        "opti_oignon/t_indirect.py": [("jot", "put", "_keep")],
        "opti_oignon/t_attr.py": [("jot", "drop", "_drop")],
        "opti_oignon/t_lambda.py": [("jot", "put", None)],
        "opti_oignon/t_tried.py": [("jot", "put", "write_tried")],
    }, sites


def test_sw26_the_callers_rule_follows_every_reference_and_only_those():
    returned = _DECIDED.replace("def accept(x):\n    return _private(x)\n", "def accept(x):\n    return _private\n")
    param = ("import opti_oignon.a as a_mod\nfrom opti_oignon.jot import get_jot\n\nm = get_jot()\n\n\n"
             "def run(m, x):\n    return m.apply(x)\n\n\ndef go(x):\n    return run(a_mod, x)\n")
    other = {"opti_oignon/other.py": "def apply(x):\n    return x\n",
             "opti_oignon/f2.py": "import opti_oignon.other as other\n\n\ndef g():\n    return other.apply('z')\n"}
    lazy = {"opti_oignon/pk/__init__.py": '_EXPORTS = {"run_lazy": ("opti_oignon.a", "apply")}\n',
            "opti_oignon/lazy_caller.py": "from opti_oignon.pk import run_lazy\n\n\ndef f():\n    run_lazy('z')\n"}
    method = ("from opti_oignon.gate import allow\nfrom opti_oignon.jot import get_jot\n\n\nclass K:\n"
              "    def _w(self, x):\n        get_jot().put(x)\n\n    def go(self, x):\n        if allow(x):\n"
              "            self._w(x)\n")
    aliased_method = method + "\n    alias = _w\n"
    guard, restore = _load()
    try:
        gates = {"keeper": guard.Gate("decision", "opti_oignon/a.py", "Gatekeeper.write", (), "a test gate"),
                 "accept": guard.Gate("decision", "opti_oignon/a.py", "accept", (), "a test gate")}
        _tables(guard, gates=gates, gated={("opti_oignon/a.py", "jot"): ("keeper", "accept")})

        def refused(modules):
            found = guard.find_ungated(_census(guard, modules))
            return sorted(re.search(r" in ([\w.]+)\(\)", line).group(1) for line in found)

        returning = refused({"opti_oignon/a.py": returned})
        handed_param = refused({"opti_oignon/a.py": _DECIDED, "opti_oignon/e.py": param})
        elsewhere = refused({"opti_oignon/a.py": _DECIDED, **other})
        lazily = refused({"opti_oignon/a.py": _DECIDED, **lazy})
        allow_gate = {"allow": guard.Gate("verdict", _GATE_HOME, "allow", (), "a test gate")}
        _tables(guard, gates=allow_gate, gated={("opti_oignon/k.py", "jot"): ("allow",)})
        method_alone = refused({"opti_oignon/k.py": method, _GATE_HOME: _GATE_TEXT})
        method_aliased = refused({"opti_oignon/k.py": aliased_method, _GATE_HOME: _GATE_TEXT})
    finally:
        restore()
    assert returning == ["_private"], returning
    assert handed_param == ["apply"], handed_param
    assert elsewhere == [], elsewhere
    assert lazily == ["apply"], lazily
    assert method_alone == [], method_alone
    assert method_aliased == ["K._w"], method_aliased


def test_sw27_an_authority_reaches_through_helpers_and_a_house_covers_only_its_writes():
    helper = ("from opti_oignon.jot import get_jot\n\n\ndef _write_back(x):\n    get_jot().put(x)\n\n\n"
              "def caption(x, approve=False):\n    if not approve:\n        return None\n    _write_back(x)\n")
    web = ("from fastapi import APIRouter\n\nfrom opti_oignon.approve import caption\n\nrouter = APIRouter()\n\n\n"
           "@router.post('/caption')\ndef route(x, approve: bool = False):\n    return caption(x, approve=approve)\n")
    model = "from opti_oignon.approve import caption\n\n\ndef tool(x):\n    caption(x, approve=True)\n"
    guarded = _JOT_TEXT.replace("    def put(self, text, context=None):\n",
                                "    def put(self, text, context=None):\n        self._check(text)\n").replace(
        "    def drop(self, key):\n", "    def drop(self, key):\n        self._check(key)\n")
    around = "from opti_oignon.jot import get_jot\n\n\ndef f(x):\n    get_jot()._rows.append(x)\n    get_jot().put(x)\n"
    guard, restore = _load()
    try:
        _tables(guard, gates={"approved": guard.Gate("approval", _APPROVAL_HOME, "approve", (), "r")},
                gated={(_APPROVAL_HOME, "jot"): ("approved",)})
        from_route = guard.find_ungated(_census(guard, {_APPROVAL_HOME: helper, "opti_oignon/web.py": web}))
        from_model = guard.find_ungated(_census(guard, {_APPROVAL_HOME: helper, "opti_oignon/web.py": web,
                                                        "opti_oignon/model.py": model}))
        _tables(guard, gates={"own": guard.Gate("house", _JOT, "_check", (), "r", ("opti_oignon/h.py",))},
                gated={("opti_oignon/h.py", "jot"): ("own",)})
        housed = guard.find_ungated(guard.take_census(guard.Estate(
            Path("/nonexistent-write-census"),
            {"opti_oignon/__init__.py": "", _JOT: guarded, "opti_oignon/h.py": around}, [])))
    finally:
        restore()
    assert from_route == [], from_route
    assert from_model == [f"{_APPROVAL_HOME}:{_line(helper, 'put(')}: a jot write whose approval gate (approved) "
                          f"takes its authority from a caller that no route or named entry alone reaches"], from_model
    assert housed == [f"opti_oignon/h.py:{_line(around, '_rows')}: a jot write (_rows) in f() that none of its "
                      f"gates (own) covers"], housed


def test_sw28_a_reason_in_prose_lists_its_sites_and_a_constant_binds_no_context():
    two = ("from opti_oignon.jot import get_jot\n\n\ndef f():\n    get_jot().put('a')\n\n\n"
           "def g():\n    get_jot().put('b')\n")
    # Five constants, the last a module constant; a parameter is the control.
    constant = ("from opti_oignon.jot import get_jot\n\nSAME = 'always'\n\n\ndef f(c):\n"
                "    get_jot().put('a', context='always-the-same')\n    get_jot().put('b', context=SAME)\n"
                "    get_jot().put('c', context=f'same')\n    get_jot().put('d', context=('s',))\n"
                "    get_jot().put('e', context=-1)\n    get_jot().put('f', context=c)\n")
    a_jot, k_jot = ("opti_oignon/a.py", "jot"), ("opti_oignon/k.py", "jot")
    guard, restore = _load()
    try:
        exempt = {a_jot: ("copy", "r"), k_jot: ("keyed", "r")}
        modules = {"opti_oignon/a.py": two, "opti_oignon/k.py": constant}
        _tables(guard, exempt=exempt, argued={a_jot: (("f", "put"),)})
        over = guard.find_failed_exemptions(_census(guard, modules))
        _tables(guard, exempt=exempt, argued={a_jot: (("f", "put"), ("g", "put"), ("h", "put"))})
        under = guard.find_stale_entries(_census(guard, modules))
        _tables(guard, exempt=exempt)
        uncounted = guard.find_failed_exemptions(_census(guard, modules))
        z_jot = ("opti_oignon/z.py", "jot")
        _tables(guard, exempt=exempt, argued={a_jot: (("f", "put"), ("g", "put")), z_jot: (("f", "put"),)})
        misplaced = guard.find_stale_entries(_census(guard, modules))
        # A key predicate's site argued apart absorbs one refusal; argued past what the predicate refuses, stale.
        _tables(guard, exempt=exempt, argued={k_jot: (("f", "put"),)})
        apart = [line for line in guard.find_failed_exemptions(_census(guard, modules))
                 if line.startswith("opti_oignon/k.py")]
        _tables(guard, exempt=exempt, argued={k_jot: (("f", "put"),) * 6})
        overargued = guard.find_stale_entries(_census(guard, modules))
        # The same number of sites, one swapped for another.
        _tables(guard, exempt=exempt, argued={a_jot: (("f", "put"), ("h", "put"))})
        census = _census(guard, modules)
        swapped = guard.find_failed_exemptions(census) + guard.find_stale_entries(census)
    finally:
        restore()
    keyed = [f"opti_oignon/k.py:{_line(constant, needle)}: a jot write that binds no context key (context)"
             for needle in ("put('a'", "put('b'", "put('c'", "put('d'", "put('e'")]
    unlisted = "opti_oignon/a.py: a jot write (put) in g() that its reason was not written for"
    gone = "opti_oignon/a.py: its reason argues for a jot write (put) in h() the census no longer finds; take it off"
    assert sorted(over) == sorted(keyed + [unlisted]), over
    assert under == [gone], under
    assert sorted(uncounted) == sorted(
        keyed + ["opti_oignon/a.py: jot is exempt by a reason alone and names none of the sites it covers"]), uncounted
    assert misplaced == ["opti_oignon/z.py: sites argued for jot, which no reason in prose exempts; take them off"], \
        misplaced
    assert len(apart) == len(keyed) - 1 and set(apart) < set(keyed), apart
    assert overargued == ["opti_oignon/k.py: its reason argues apart 6 jot write(s) its key predicate would refuse, "
                          "and the predicate refuses 5 of them; take the others off"], overargued
    assert sorted(swapped) == sorted(keyed + [unlisted, gone]), swapped


_DOMINANCE_TWO = '''from opti_oignon.gate import allow
from opti_oignon.jot import get_jot


def elsewise(x):
    if not allow(x):
        return None
    else:
        get_jot().put(x)


def passing(x):
    if allow(x):
        pass
    else:
        return None
    get_jot().put(x)


def flipped(x):
    ok = allow(x)

    def flip():
        nonlocal ok
        ok = True
    flip()
    if not ok:
        return None
    get_jot().put(x)


def nested_shadow(x):
    ok = allow(x)

    def inner(ok):
        if not ok:
            return None
        get_jot().put(ok)
    return inner(True)
'''


def test_sw29_a_gate_holds_in_its_else_and_never_through_a_rebinding_scope():
    guard, restore = _load()
    try:
        _gate_tables(guard, "opti_oignon/a.py", ("allow",))
        found = guard.find_ungated(_census(guard, {"opti_oignon/a.py": _DOMINANCE_TWO, _GATE_HOME: _GATE_TEXT}))
    finally:
        restore()
    refused = sorted(re.search(r" in ([\w.]+)\(\)", line).group(1) for line in found)
    assert refused == ["flipped", "nested_shadow.inner"], found


def test_sw30_a_proof_must_run_and_a_nesting_too_deep_fails_by_name(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    suites = {
        "test_ign_contracts.py": "import opti_oignon.gate\n\n\ndef test_i1_x():\n    pass\n",
        "test_des_contracts.py": "import opti_oignon.gate\n\n\ndef test_d1_x():\n    pass\n",
        "test_cls_contracts.py": ("import opti_oignon.gate\n\n\nclass Helper:\n    def test_c1_x(self):\n        pass\n\n\n"
                                  "class TestReal:\n    def test_c2_x(self):\n        pass\n"),
        "test_skip_contracts.py": "import opti_oignon.gate\nimport pytest\n\n\n@pytest.mark.skip\ndef test_s1_x():\n    pass\n",
        "test_doc_contracts.py": '"""About opti_oignon.gate."""\n\n\ndef test_o1_x():\n    pass\n',
    }
    for name, text in suites.items():
        (tests / name).write_text(text, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = """\n    --ignore=tests/test_ign_contracts.py\n'
        '    --deselect=tests/test_des_contracts.py::test_d1_x\n"""\n', encoding="utf-8")
    deep = tmp_path / "deep"
    _write_tree(deep, {_JOT: _JOT_TEXT, "opti_oignon/deep.py": (
        "from opti_oignon.jot import get_jot\n\nx = get_jot()" + ".copy()" * 1200 + "\n")})
    guard, restore = _load()
    try:
        gates = {name: guard.Gate("verdict", _GATE_HOME, "allow", (contract,), "r")
                 for name, contract in (("ignored", "i1"), ("deselected", "d1"), ("helper", "c1"), ("real", "c2"),
                                        ("skipped", "s1"), ("documented", "o1"))}
        _tables(guard, gates=gates)
        proofs = guard.find_gate_proofs_missing(_census(guard, {_GATE_HOME: _GATE_TEXT}, root=tmp_path))
        _tables(guard)
        too_deep = guard.run(deep)
    finally:
        restore()
    assert sorted(proofs) == sorted([
        "gate ignored: contract i1 names no test function that runs",
        "gate deselected: contract d1 names no test function that runs",
        "gate helper: contract c1 names no test function that runs",
        "gate skipped: contract s1 names no test function that runs",
        "gate documented: the suite of contract o1 (tests/test_doc_contracts.py) never names opti_oignon/gate.py",
    ]), proofs
    assert too_deep.code == 1 and "nests too deep" in too_deep.lines[0], too_deep.lines


# ---------------------------------------------------------------------------
# SW31 - SW42: what the third review found.
# ---------------------------------------------------------------------------
def _decided_gates(guard):
    return {"keeper": guard.Gate("decision", "opti_oignon/a.py", "Gatekeeper.write", (), "a test gate"),
            "accept": guard.Gate("decision", "opti_oignon/a.py", "accept", (), "a test gate")}


def _ungated_functions(guard, modules):
    found = guard.find_ungated(_census(guard, modules))
    return sorted(re.search(r" in (\w+)\(\)", line).group(1) for line in found)


def _modules_named(lines):
    return sorted({re.match(r"opti_oignon/(\w+)\.py", line).group(1) for line in lines})


def test_sw31_a_receiver_bound_by_anything_but_an_import_may_be_the_module():
    head = "import opti_oignon.other as m\nimport opti_oignon.a as a_mod\n"
    forms = {
        "parameter": head + "\n\ndef run(m, x):\n    return m.apply(x)\n\n\ndef go(x):\n    run(a_mod, x)\n",
        "lambda": head + "\nRUN = lambda m, x: m.apply(x)\n",
        "loop": head + "\n\ndef f(x):\n    for m in (a_mod,):\n        m.apply(x)\n",
        "rebound": ("import opti_oignon.a as a_mod\nfrom opti_oignon.jot import get_jot\n\nm = get_jot()\n\n\n"
                    "def f(x):\n    m = a_mod\n    m.apply(x)\n"),
        "attribute": ("import importlib\nimport opti_oignon.a as a_mod\nfrom opti_oignon.jot import get_jot\n\n\n"
                      "class H:\n    def __init__(self):\n        self.m = get_jot()\n\n    def swap(self):\n"
                      "        self.m = importlib.import_module('opti_oignon.a')\n\n    def go(self, x):\n"
                      "        self.m.apply(x)\n"),
    }
    # The control: a name only ever bound to a store object is no module.
    store_only = ("import opti_oignon.a as a_mod\nfrom opti_oignon.jot import get_jot\n\nm = get_jot()\n\n\n"
                  "def f(x):\n    m.apply(x)\n")
    guard, restore = _load()
    try:
        _tables(guard, gates=_decided_gates(guard), gated={("opti_oignon/a.py", "jot"): ("keeper", "accept")})
        base = {"opti_oignon/a.py": _DECIDED, "opti_oignon/other.py": ""}
        found = {key: _ungated_functions(guard, {**base, "opti_oignon/e.py": text}) for key, text in forms.items()}
        control = _ungated_functions(guard, {**base, "opti_oignon/e.py": store_only})
    finally:
        restore()
    assert found == {key: ["apply"] for key in forms}, found
    assert control == [], control
    # A nested function declared global binds a module-level name: every call of that name, in the module, is a
    # reference to it, and not only the calls inside the function that defines it.
    nested_global = ("from opti_oignon.gate import allow\nfrom opti_oignon.jot import get_jot\n\n\ndef setup():\n"
                     "    global writer\n\n    def writer(x):\n        get_jot().put(x)\n    if allow(1):\n"
                     "        writer(1)\n\n\ndef go(x):\n    writer(x)\n")
    guard, restore = _load()
    try:
        _gate_tables(guard, "opti_oignon/e.py", ("allow",))
        nested = guard.find_ungated(_census(guard, {_GATE_HOME: _GATE_TEXT, "opti_oignon/e.py": nested_global}))
    finally:
        restore()
    assert nested == [f"opti_oignon/e.py:{_line(nested_global, 'get_jot().put(')}: a jot write (put) in "
                      f"setup.writer() that none of its gates (allow) covers"], nested


def test_sw32_a_decision_hands_out_no_writer():
    base = _DECIDED.replace("def accept(x):\n    return _private(x)\n", "")
    handed = {
        "returned_partial": "import functools\n\n\ndef accept(x):\n    return functools.partial(_private, x)\n",
        "kept": "PENDING = []\n\n\ndef accept(x):\n    PENDING.append(_private)\n",
        "nested": "def accept(x):\n    def later():\n        _private(x)\n    return later\n",
        "lambda": "def accept(x):\n    return lambda: _private(x)\n",
        "generator": "def accept(xs):\n    return (_private(x) for x in xs)\n",
        "foreign_partial": "from mylib import partial\n\n\ndef accept(x):\n    return partial(_private, x)()\n",
        "named_then_handed": ("from functools import partial\n\n\ndef accept(x):\n    run = partial(_private, x)\n"
                              "    return run\n"),
        "named_twice": ("from functools import partial\n\n\ndef accept(x):\n    run = partial(_private, x)\n"
                        "    run = partial(_private, x)\n    return run()\n"),
        "built_in_nested": ("from functools import partial\n\n\ndef accept(x):\n    def later():\n"
                            "        return partial(_private, x)()\n    return later\n"),
        "second_argument": "from functools import partial\n\n\ndef accept(x):\n    return partial(print, _private)()\n",
        "two_targets": ("from functools import partial\n\n\ndef accept(x):\n    run = kept = partial(_private, x)\n"
                        "    run()\n    return kept\n"),
        "rebound_partial": ("from functools import partial\n\npartial = print\n\n\ndef accept(x):\n"
                            "    return partial(_private, x)()\n"),
    }
    in_place = {
        "called_at_once": "from functools import partial\n\n\ndef accept(x):\n    return partial(_private, x)()\n",
        "called_by_name": ("from functools import partial\n\n\ndef accept(x):\n    run = partial(_private, x)\n"
                           "    if x:\n        return run()\n    return run()\n"),
        "value_kept": "def accept(x):\n    y = _private(x)\n    return y\n",
    }
    guard, restore = _load()
    try:
        _tables(guard, gates=_decided_gates(guard), gated={("opti_oignon/a.py", "jot"): ("keeper", "accept")})
        refused = {key: _ungated_functions(guard, {"opti_oignon/a.py": base + "\n\n" + text})
                   for key, text in {**handed, **in_place}.items()}
    finally:
        restore()
    assert refused == {**{key: ["_private"] for key in handed}, **{key: [] for key in in_place}}, refused


def test_sw33_a_verdict_covers_no_nested_function_or_lambda_and_a_filter_follows_its_data():
    head = "import threading\n\nfrom opti_oignon.gate import allow, pick\nfrom opti_oignon.jot import get_jot\n\n\n"
    refused_forms = {
        "nested": (head + "def f(x):\n    if not allow(x):\n        return None\n\n    def w(y):\n"
                   "        get_jot().put(y)\n    return w\n"),
        "lambda": head + "def f(x):\n    if not allow(x):\n        return None\n    return lambda y: get_jot().put(y)\n",
        "shadowed": head + "def f(messages):\n    kept = pick(messages)\n    return lambda kept: get_jot().put(kept)\n",
        "lambda_in_helper": (head + "def helper():\n    return lambda y: get_jot().put(y)\n\n\ndef f(x):\n"
                             "    if not allow(x):\n        return None\n    return helper()\n"),
    }
    gated_forms = {
        "called": (head + "def f(x):\n    if not allow(x):\n        return None\n\n    def w():\n"
                   "        get_jot().put(x)\n    w()\n"),
        "closure": (head + "def f(messages):\n    kept = pick(messages)\n\n    def w():\n        get_jot().put(kept)\n"
                    "    threading.Thread(target=w).start()\n"),
    }
    forms = {**refused_forms, **gated_forms}
    guard, restore = _load()
    try:
        defined = {"allow": guard.Gate("verdict", _GATE_HOME, "allow", (), "r"),
                   "pick": guard.Gate("filter", _GATE_HOME, "pick", (), "r")}
        _tables(guard, gates=defined, gated={(f"opti_oignon/{key}.py", "jot"): ("allow", "pick") for key in forms})
        found = guard.find_ungated(_census(guard, {
            _GATE_HOME: _GATE_TEXT, **{f"opti_oignon/{key}.py": text for key, text in forms.items()}}))
    finally:
        restore()
    assert _modules_named(found) == sorted(refused_forms), found


def test_sw34_an_instance_is_placed_by_a_closed_grammar_through_names_bound_to_nothing_else():
    imports = "import os\nimport tempfile\nfrom pathlib import Path\n\nfrom opti_oignon.jot import Jot\n\n\n"
    refused_forms = {
        "either": "def f(user_path):\n    Jot(path=user_path or tempfile.mkdtemp()).put('a')\n",
        "joined_name": ("def f(name):\n    with tempfile.TemporaryDirectory() as tmp:\n"
                        "        Jot(path=Path(tmp) / name).put('a')\n"),
        "contains": ("HOME = '/home/user'\n\n\ndef f():\n"
                     "    Jot(path=os.path.join(HOME, os.path.basename(tempfile.mkdtemp()))).put('a')\n"),
        "walrus": ("def f(user_path):\n    tmp = tempfile.mkdtemp()\n    if (tmp := user_path):\n"
                   "        Jot(path=tmp).put('a')\n"),
        "augmented": "def f(user_db):\n    tmp = tempfile.mkdtemp()\n    tmp += user_db\n    Jot(path=tmp).put('a')\n",
        "pattern": ("def f(user_path):\n    tmp = tempfile.mkdtemp()\n    match user_path:\n        case tmp:\n"
                    "            pass\n    Jot(path=tmp).put('a')\n"),
        "elsewhere": ("def a():\n    tmp = tempfile.mkdtemp()\n    Jot(path=tmp).put('a')\n\n\n"
                      "def b(user):\n    tmp = user\n    return tmp\n"),
        "parameter": "def a():\n    tmp = tempfile.mkdtemp()\n    return tmp\n\n\ndef b(tmp):\n    Jot(path=tmp).put('a')\n",
        "own_parameter": ("def a():\n    j = Jot(path=tempfile.mkdtemp())\n    j.put('a')\n\n\n"
                          "def b(j):\n    j.put('b')\n"),
        "positional": "def f():\n    Jot(tempfile.mkdtemp()).put('a')\n",
        "subclass": "class Mine(Jot):\n    pass\n\n\ndef f():\n    Mine(path=tempfile.mkdtemp()).put('a')\n",
        "shared_root": "def f():\n    Jot(path=os.path.join(tempfile.gettempdir(), 'jot.db')).put('a')\n",
        "climbing": "def f():\n    tmp = tempfile.mkdtemp()\n    Jot(path=os.path.join(tmp, '../jot.db')).put('a')\n",
        "absolute": "def f():\n    tmp = tempfile.mkdtemp()\n    Jot(path=Path(tmp) / '/home/user/jot.db').put('a')\n",
        "fake_tempfile": "def mkdtemp():\n    return '/home/user'\n\n\ndef f():\n    Jot(path=mkdtemp()).put('a')\n",
        "shadowed_str": ("def str(x):\n    return '/home/user/jot.db'\n\n\ndef f():\n    tmp = tempfile.mkdtemp()\n"
                         "    Jot(path=str(tmp)).put('a')\n"),
    }
    placed_forms = {
        "joined_constant": ("def f():\n    with tempfile.TemporaryDirectory() as tmp:\n"
                            "        Jot(path=Path(tmp) / 'j.db').put('a')\n"),
        "joined": "def f():\n    tmp = tempfile.mkdtemp()\n    Jot(path=os.path.join(tmp, 'j.db')).put('a')\n",
        "called": "def f():\n    Jot(path=tempfile.mkdtemp(prefix='x')).put('a')\n",
        "named": "def f():\n    tmp = Path(tempfile.mkdtemp())\n    Jot(path=tmp).put('a')\n",
        "own": "def f():\n    j = Jot(path=tempfile.mkdtemp())\n    j.put('a')\n",
    }
    forms = {**refused_forms, **placed_forms}
    guard, restore = _load()
    try:
        _tables(guard, exempt={(f"opti_oignon/{key}.py", "jot"): ("instance", "r") for key in forms})
        failed = guard.find_failed_exemptions(_census(guard, {
            f"opti_oignon/{key}.py": imports + text for key, text in forms.items()}))
    finally:
        restore()
    assert _modules_named(failed) == sorted(refused_forms), failed


def test_sw35_an_expression_names_every_module_it_may():
    shared = {"opti_oignon/helper_b.py": "", "opti_oignon/pkg/__init__.py": "",
              "opti_oignon/pkg/impl.py": "def write_it(store, x):\n    store.put(x)\n"}
    either = ("import importlib\nimport sys\n\nfrom opti_oignon.jot import get_jot\n\n\ndef f():\n"
              "    m = sys.modules.get('opti_oignon.helper_b') or importlib.import_module('opti_oignon.pkg')\n"
              "    m.impl.write_it(get_jot(), 'x')\n")
    tried = ("from opti_oignon.jot import get_jot\n\ntry:\n    import opti_oignon.{0} as m\nexcept ImportError:\n"
             "    import opti_oignon.{1} as m\n\n\ndef f():\n    m.impl.write_it(get_jot(), 'x')\n")
    held = ("import sys\n\n\nclass H:\n    def a(self):\n        self.mod = sys.modules.get('opti_oignon.helper_b')\n\n"
            "    def b(self):\n        self.mod = sys.modules.get('opti_oignon.jot')\n\n"
            "    def c(self):\n        self.mod.put_line('x')\n")
    guard, restore = _load()
    try:
        _tables(guard)
        runs = {"either": _sites(_census(guard, {**shared, "opti_oignon/e.py": either})),
                "tried": _sites(_census(guard, {**shared, "opti_oignon/e.py": tried.format("helper_b", "pkg")})),
                "swapped": _sites(_census(guard, {**shared, "opti_oignon/e.py": tried.format("pkg", "helper_b")}))}
        attribute = _sites(_census(guard, {**shared, "opti_oignon/h.py": held}))
    finally:
        restore()
    impl = [("jot", "put", "write_it")]
    assert {key: sites.get("opti_oignon/pkg/impl.py") for key, sites in runs.items()} == {
        "either": impl, "tried": impl, "swapped": impl}, runs
    assert attribute.get("opti_oignon/h.py") == [("jot", "put_line", "H.c")], attribute


def test_sw36_a_gate_is_its_own_name_bound_once_or_its_homes_attribute():
    head = "import importlib\n\nfrom opti_oignon.jot import get_jot\n"
    refused_forms = {
        "rebound_module": (head + "from opti_oignon import gate as g\n\n\ndef f(x):\n"
                           "    g = importlib.import_module('opti_oignon.fake')\n    if not g.allow(x):\n"
                           "        return None\n    get_jot().put(x)\n"),
        "rebound_name": (head + "from opti_oignon.gate import allow\n\nallow = lambda y: True\n\n\ndef f(x):\n"
                         "    if not allow(x):\n        return None\n    get_jot().put(x)\n"),
        "captured": (head + "from opti_oignon.gate import allow\n\n\ndef f(x, y):\n    ok = allow(x)\n"
                     "    match y:\n        case ok:\n            pass\n    if not ok:\n        return None\n"
                     "    get_jot().put(x)\n"),
    }
    gated_forms = {
        "by_name": (head + "from opti_oignon.gate import allow\n\n\ndef f(x):\n    if not allow(x):\n"
                    "        return None\n    get_jot().put(x)\n"),
        "by_module": (head + "from opti_oignon import gate as g\n\n\ndef f(x):\n    if not g.allow(x):\n"
                      "        return None\n    get_jot().put(x)\n"),
    }
    forms = {**refused_forms, **gated_forms}
    guard, restore = _load()
    try:
        _tables(guard, gates={"allow": guard.Gate("verdict", _GATE_HOME, "allow", (), "r")},
                gated={(f"opti_oignon/{key}.py", "jot"): ("allow",) for key in forms})
        found = guard.find_ungated(_census(guard, {
            _GATE_HOME: _GATE_TEXT, "opti_oignon/fake.py": "def allow(x):\n    return True\n",
            **{f"opti_oignon/{key}.py": text for key, text in forms.items()}}))
    finally:
        restore()
    assert _modules_named(found) == sorted(refused_forms), found


def test_sw37_a_module_imported_by_keyword_by_a_list_of_names_or_through_its_package_is_followed():
    forms = {
        "keyword": ("import importlib\n\n\ndef f():\n    m = importlib.import_module('.jot', package='opti_oignon')\n"
                    "    m.get_jot().put('x')\n"),
        "named": ("import importlib\n\n\ndef f():\n"
                  "    m = importlib.import_module(name='.jot', package='opti_oignon')\n    m.get_jot().put('x')\n"),
        "swapped": ("import importlib\n\n\ndef f():\n"
                    "    m = importlib.import_module(package='opti_oignon', name='.jot')\n    m.get_jot().put('x')\n"),
        "fromlist": ("NAMES = ['put_line']\n\n\ndef f():\n"
                     "    m = __import__('opti_oignon.jot', globals(), locals(), NAMES)\n    m.put_line('x')\n"),
        "chain": "from opti_oignon import pkg\n\n\ndef f():\n    pkg.inner.get_jot().put('x')\n",
        "dotted": ("import importlib\n\n\ndef f():\n    m = importlib.import_module('opti_oignon.pkg')\n"
                   "    m.inner.get_jot().put('x')\n"),
    }
    shared = {"opti_oignon/pkg/__init__.py": "", "opti_oignon/pkg/inner.py": "from opti_oignon.jot import get_jot\n"}
    guard, restore = _load()
    try:
        _tables(guard)
        sites = _sites(_census(guard, {**shared, **{f"opti_oignon/{key}.py": text for key, text in forms.items()}}))
    finally:
        restore()
    put = [("jot", "put", "f")]
    assert {key: sites.get(f"opti_oignon/{key}.py") for key in forms} == {
        "keyword": put, "named": put, "swapped": put, "fromlist": [("jot", "put_line", "f")], "chain": put,
        "dotted": put}, sites


def test_sw38_a_dispatch_table_of_bound_methods_and_other_modules_functions_is_followed():
    # A parameter name of its own for each function: names are not scoped, and one shared name bound through any
    # one route would type all three.
    helper = ("def add(added, x):\n    added.put(x)\n\n\ndef keep(kept, x):\n    kept.put(x)\n\n\n"
              "class Other:\n    def write(self, written, x):\n        written.put(x)\n")
    tables = ("import opti_oignon.helper as helper\nfrom opti_oignon.helper import Other, keep\n"
              "from opti_oignon.jot import get_jot\n\n\nclass Box:\n    def _remove(self, store, x):\n"
              "        store.drop(x)\n\n    def handle(self, x):\n        table = {'remove': self._remove}\n"
              "        return table['remove'](get_jot(), x)\n\n\nBY_MODULE = {'add': helper.add}\n"
              "BY_NAME = {'keep': keep}\nBY_OBJECT = {'write': Other().write}\n\n\ndef handle(action, x):\n"
              "    BY_MODULE.get(action)(get_jot(), x)\n    BY_OBJECT[action](get_jot(), x)\n"
              "    return BY_NAME[action](get_jot(), x)\n")
    guard, restore = _load()
    try:
        _tables(guard)
        sites = _sites(_census(guard, {"opti_oignon/helper.py": helper, "opti_oignon/tables.py": tables}))
    finally:
        restore()
    assert sites.get("opti_oignon/tables.py") == [("jot", "drop", "Box._remove")], sites
    assert sites.get("opti_oignon/helper.py") == [("jot", "put", "Other.write"), ("jot", "put", "add"),
                                                  ("jot", "put", "keep")], sites


def test_sw39_a_proof_is_read_through_classes_marks_and_the_selection_rule(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    suites = {
        "test_cls_contracts.py": ("import opti_oignon.gate\n\n\nclass TestReal:\n    def test_k1_x(self):\n"
                                  "        pass\n\n    def test_k2_x(self):\n        pass\n\n\nclass TestOther:\n"
                                  "    def test_k3_x(self):\n        pass\n"),
        "test_mark_contracts.py": ("import opti_oignon.gate\nimport pytest\n\npytestmark = [pytest.mark.skip]\n\n\n"
                                   "def test_m1_x():\n    pass\n"),
        "test_init_contracts.py": ("import opti_oignon.gate\n\n\nclass TestBuilt:\n    def __init__(self):\n"
                                   "        pass\n\n    def test_n1_x(self):\n        pass\n"),
        "test_off_contracts.py": ("import opti_oignon.gate\n\n\nclass TestOff:\n    __test__ = False\n\n"
                                  "    def test_f1_x(self):\n        pass\n"),
        "test_self_contracts.py": ("import opti_oignon.gate\nimport pytest\n\npytest.importorskip('nothing_here')\n\n\n"
                                   "def test_q1_x():\n    pass\n"),
    }
    for name, text in suites.items():
        (tests / name).write_text(text, encoding="utf-8")
    rule = ('[tool.pytest.ini_options]\naddopts = """\n    --deselect=tests/test_cls_contracts.py::TestReal::test_k1_x\n'
            '    --deselect=tests/test_cls_contracts.py::TestOther\n"""\n')
    (tmp_path / "pyproject.toml").write_text(rule, encoding="utf-8")
    guard, restore = _load()
    try:
        gates = {name: guard.Gate("verdict", _GATE_HOME, "allow", (contract,), "r")
                 for name, contract in (("by_name", "k1"), ("kept", "k2"), ("by_class", "k3"), ("marked", "m1"),
                                        ("built", "n1"), ("off", "f1"), ("self_skipped", "q1"))}
        _tables(guard, gates=gates)
        proofs = guard.find_gate_proofs_missing(_census(guard, {_GATE_HOME: _GATE_TEXT}, root=tmp_path))
        (tmp_path / "pyproject.toml").write_text(rule.replace('"""\n', '"""\n    -k real\n', 1), encoding="utf-8")
        unread = guard.find_gate_proofs_missing(_census(guard, {_GATE_HOME: _GATE_TEXT}, root=tmp_path))
        (tmp_path / "pyproject.toml").write_text(
            '[tool.pytest.ini_options]\naddopts = """\n    --ignore=tests/\n"""\n', encoding="utf-8")
        directory = guard.find_gate_proofs_missing(_census(guard, {_GATE_HOME: _GATE_TEXT}, root=tmp_path))
    finally:
        restore()
    assert sorted(directory) == sorted(f"gate {gate}: contract {contract} names no test function that runs"
                                       for gate, contract in (("by_name", "k1"), ("kept", "k2"), ("by_class", "k3"),
                                                              ("marked", "m1"), ("built", "n1"), ("off", "f1"),
                                                              ("self_skipped", "q1"))), directory
    runs = "names no test function that runs"
    assert sorted(proofs) == sorted(f"gate {gate}: contract {contract} {runs}" for gate, contract in (
        ("by_name", "k1"), ("by_class", "k3"), ("marked", "m1"), ("built", "n1"), ("off", "f1"),
        ("self_skipped", "q1"))), proofs
    assert sorted(set(unread) - set(proofs)) == [
        "the selection rule holds what the census cannot read (-k real): no gate proof can be read under it"
    ], unread


def test_sw40_a_house_gate_covers_its_methods_on_store_objects_and_binds_their_overrides():
    house = _JOT_TEXT.replace("    def put(self, text, context=None):\n",
                              "    def put(self, text, context=None):\n        self._check(text)\n").replace(
        "    def drop(self, key):\n", "    def drop(self, key):\n        self._check(key)\n") + (
        "\n\ndef put(text):\n    return text\n")
    caller = "from opti_oignon.jot import get_jot, put\n\n\ndef f(x):\n    get_jot().put(x)\n    put(x)\n"
    loud = ("from opti_oignon.jot import Jot\n\n\nclass Loud(Jot):\n    def put(self, text, context=None):\n"
            "        self.rows = [text]\n        return text\n")
    quiet = ("from opti_oignon.jot import Jot\n\n\nclass Quiet(Jot):\n    def put(self, text, context=None):\n"
             "        return super().put(text, context)\n")
    guard, restore = _load()
    try:
        gates = {"own": guard.Gate("house", _JOT, "_check", (), "r", ("opti_oignon/h.py",))}
        _tables(guard, stores={"jot": _jot(guard, functions=("put_line", "put"))}, gates=gates,
                gated={("opti_oignon/h.py", "jot"): ("own",)})
        census = guard.take_census(guard.Estate(Path("/nonexistent-write-census"), {
            "opti_oignon/__init__.py": "", _JOT: house, "opti_oignon/h.py": caller, "opti_oignon/loud.py": loud,
            "opti_oignon/quiet.py": quiet}, []))
        ungated = guard.find_ungated(census)
        drift = guard.find_table_drift(census)
    finally:
        restore()
    assert ungated == [f"opti_oignon/h.py:{_line(caller, '    put(x)')}: a jot write (put) in f() that none of its "
                       f"gates (own) covers"], ungated
    assert drift == ["gate own: opti_oignon/loud.py: Loud.put overrides a write of jot and writes before it calls "
                     "_check"], drift


def test_sw41_a_house_function_defined_under_a_module_level_try_is_found():
    tried = _JOT_TEXT.replace("def get_jot():\n    return Jot()\n",
                              "try:\n    def get_jot():\n        return Jot()\nexcept ImportError:\n    get_jot = None\n")
    gone = _JOT_TEXT.replace("def get_jot():\n    return Jot()\n", "")
    guard, restore = _load()
    try:
        _tables(guard)
        drift = {key: guard.find_table_drift(guard.take_census(guard.Estate(
            Path("/nonexistent-write-census"), {"opti_oignon/__init__.py": "", _JOT: text}, [])))
            for key, text in (("tried", tried), ("gone", gone))}
    finally:
        restore()
    assert drift == {"tried": [], "gone": [f"store jot: {_JOT} defines no function get_jot"]}, drift


def test_sw42_a_site_another_gate_of_its_pair_covers_needs_no_authority():
    both = ("from opti_oignon.gate import allow\nfrom opti_oignon.jot import get_jot\n\n\n"
            "def caption(x, approve=False):\n    if not allow(x):\n        return None\n    get_jot().put(x)\n\n\n"
            "def bare(x, approve=False):\n    if not approve:\n        return None\n    get_jot().put(x)\n")
    model = ("from opti_oignon.approve import bare, caption\n\n\n"
             "def tool(x):\n    caption(x, approve=True)\n    bare(x, approve=True)\n")
    guard, restore = _load()
    try:
        gates = {"approved": guard.Gate("approval", _APPROVAL_HOME, "approve", (), "r"),
                 "allow": guard.Gate("verdict", _GATE_HOME, "allow", (), "r")}
        _tables(guard, gates=gates, gated={(_APPROVAL_HOME, "jot"): ("allow", "approved")})
        found = guard.find_ungated(_census(guard, {
            _GATE_HOME: _GATE_TEXT, _APPROVAL_HOME: both, "opti_oignon/model.py": model}))
    finally:
        restore()
    assert found == [f"{_APPROVAL_HOME}:{_line(both, 'def bare') + 3}: a jot write whose approval gate (approved) "
                     f"takes its authority from a caller that no route or named entry alone reaches"], found


# ---------------------------------------------------------------------------
# SW43 - SW57: what the fourth review found.
# ---------------------------------------------------------------------------
_HOUSE = _JOT_TEXT.replace("    def put(self, text, context=None):\n",
                           "    def put(self, text, context=None):\n        self._check(text)\n").replace(
    "    def drop(self, key):\n", "    def drop(self, key):\n        self._check(key)\n") + (
    "\n\ndef put(text):\n    return text\n")


def _house_census(guard, modules):
    gates = {"own": guard.Gate("house", _JOT, "_check", (), "r", ("opti_oignon/h.py",))}
    _tables(guard, stores={"jot": _jot(guard, functions=("put_line", "put"))}, gates=gates,
            gated={("opti_oignon/h.py", "jot"): ("own",)})
    return guard.take_census(guard.Estate(Path("/nonexistent-write-census"), {
        "opti_oignon/__init__.py": "", _JOT: _HOUSE, **modules}, []))


def test_sw43_a_house_gate_holds_through_every_subclass_and_only_on_store_objects():
    head = "from opti_oignon.jot import Jot\n\n\n"
    subclasses = {
        "lax": head + "class Lax(Jot):\n    def _check(self, value):\n        return None\n",
        "aliased": head + "def raw(self, text, context=None):\n    return text\n\n\nclass Aliased(Jot):\n    put = raw\n",
        "ahead": (head + "class Raw:\n    def put(self, text, context=None):\n        return text\n\n\n"
                  "class Ahead(Raw, Jot):\n    pass\n"),
        "skipping": (head + "class Skipping(Jot):\n    def put(self, text, context=None):\n"
                     "        return super(Jot, self).put(text, context)\n"),
        "handing": (head + "class Handing(Jot):\n    def put(self, text, context=None):\n"
                    "        return super().put(text, context)\n"),
    }
    caller = ("import opti_oignon.jot as jm\nfrom opti_oignon.jot import get_jot\n\n\n"
              "def f(x):\n    get_jot().put(x)\n    jm.put(x)\n")
    guard, restore = _load()
    try:
        census = _house_census(guard, {"opti_oignon/h.py": caller,
                                       **{f"opti_oignon/{key}.py": text for key, text in subclasses.items()}})
        drift = guard.find_table_drift(census)
        ungated = guard.find_ungated(census)
    finally:
        restore()
    assert sorted(drift) == sorted([
        "gate own: opti_oignon/lax.py: Lax binds _check, the gate of jot, anew",
        "gate own: opti_oignon/aliased.py: Aliased.put overrides a write of jot and writes before it calls _check",
        "gate own: opti_oignon/ahead.py: Ahead has a base besides jot's class, which may run before or around its "
        "writes",
        "gate own: opti_oignon/skipping.py: Skipping.put overrides a write of jot and writes before it calls _check",
    ]), drift
    assert ungated == [f"opti_oignon/h.py:{_line(caller, 'jm.put(')}: a jot write (put) in f() that none of its gates "
                       f"(own) covers"], ungated
    # Every binding of a subclass's body counts, through its blocks, an import and a loop, in a nested class too;
    # a decorator, a metaclass, a hook or a base behind the store's class may reroute its writes.
    more = {
        "branch": head + "class Branch(Jot):\n    if True:\n        def put(self, text, context=None):\n"
                         "            return text\n",
        "tried": head + ("class Tried(Jot):\n    try:\n        def put(self, text, context=None):\n"
                         "            return text\n    except Exception:\n        pass\n"),
        "looped": head + ("def raw(self, text, context=None):\n    return text\n\n\nclass Looped(Jot):\n"
                          "    for put in (raw,):\n        pass\n"),
        "imported": head + "class Imported(Jot):\n    from os.path import join as _check\n",
        "nested": head + ("def make():\n    class Inner(Jot):\n        def put(self, text, context=None):\n"
                          "            return text\n    return Inner\n"),
        "decorated": head + "def keep(cls):\n    return cls\n\n\n@keep\nclass Decorated(Jot):\n    pass\n",
        "metaclass": head + "class Metaed(Jot, metaclass=type):\n    pass\n",
        "hooked": head + ("class Hooked(Jot):\n    def __getattribute__(self, name):\n"
                          "        return object.__getattribute__(self, name)\n"),
        "mixed": head + "class Raw:\n    pass\n\n\nclass Mixed(Jot, Raw):\n    pass\n",
    }
    # Setting an attribute of the store's class, from any module, rewires every object of it.
    patched = "from opti_oignon.jot import Jot\n\nJot._check = lambda self, value: None\n"
    # A parameter may be handed the house module, whose function of the same name is no gated write.
    handed_text = ("import opti_oignon.jot as jm\nfrom opti_oignon.jot import get_jot\n\n\n"
                   "def write(s, x):\n    s.put(x)\n\n\ndef f(x):\n    write(get_jot(), x)\n    write(jm, x)\n")
    doubled = _HOUSE.replace("    def drop(self, key):\n", "    def put(self, text, context=None):\n"
                             "        return text\n\n    def drop(self, key):\n", 1)
    guard, restore = _load()
    try:
        census = _house_census(guard, {"opti_oignon/patch.py": patched,
                                       **{f"opti_oignon/{key}.py": text for key, text in more.items()}})
        more_drift = guard.find_table_drift(census)
        patched_sites = _sites(census).get("opti_oignon/patch.py")
        handed = guard.find_ungated(_house_census(guard, {"opti_oignon/h.py": handed_text}))
        # The tables stay those of the house census: the house itself binds a write twice.
        twice = guard.find_table_drift(guard.take_census(guard.Estate(Path("/nonexistent-write-census"), {
            "opti_oignon/__init__.py": "", _JOT: doubled}, [])))
    finally:
        restore()
    overrides = "overrides a write of jot and writes before it calls _check"
    rebuilds = "carries a class decorator or a metaclass, which may rebuild it"
    assert sorted(more_drift) == sorted([
        f"gate own: opti_oignon/branch.py: Branch.put {overrides}",
        f"gate own: opti_oignon/tried.py: Tried.put {overrides}",
        f"gate own: opti_oignon/looped.py: Looped.put {overrides}",
        "gate own: opti_oignon/imported.py: Imported binds _check, the gate of jot, anew",
        f"gate own: opti_oignon/nested.py: Inner.put {overrides}",
        f"gate own: opti_oignon/decorated.py: Decorated {rebuilds}",
        f"gate own: opti_oignon/metaclass.py: Metaed {rebuilds}",
        "gate own: opti_oignon/hooked.py: Hooked defines __getattribute__, which may reroute any of its methods",
        "gate own: opti_oignon/mixed.py: Mixed has a base besides jot's class, which may run before or around its "
        "writes",
    ]), more_drift
    assert patched_sites == [("jot", "_check", None)], patched_sites
    assert handed == [f"opti_oignon/h.py:{_line(handed_text, 's.put(')}: a jot write (put) in write() that none of "
                      f"its gates (own) covers"], handed
    assert twice == ["gate own: Jot.put is bound more than once in its class: only one body is gated"], twice
    # An attribute a carrier class binds to a store object is proven one: a ``setattr`` on another object does not
    # undo the proof; a ``setattr`` on an object of that class does.
    carrier = ("from opti_oignon.jot import get_jot\n\n\nclass H:\n    def __init__(self):\n        self.store = get_jot()\n\n"
               "    def go(self, x):\n        self.store.put(x)\n\n\ndef unrelated(obj, name, value):\n"
               "    setattr(obj, name, value)\n")
    rewired_carrier = carrier + "\n\ndef rewire(jm_mod):\n    h = H()\n    setattr(h, 'store', jm_mod)\n"
    guard, restore = _load()
    try:
        kept = guard.find_ungated(_house_census(guard, {"opti_oignon/h.py": carrier}))
        lost = guard.find_ungated(_house_census(guard, {"opti_oignon/h.py": rewired_carrier}))
    finally:
        restore()
    assert kept == [], kept
    assert lost == [f"opti_oignon/h.py:{_line(rewired_carrier, 'self.store.put(')}: a jot write (put) in H.go() that "
                    f"none of its gates (own) covers"], lost
    # A walrus or a match capture in a subclass's body binds a write; a class the receiver may be built from
    # besides the carrier binds the attribute to a module; ``object.__setattr__`` on the carrier rewires it.
    bound_otherwise = {
        "walrus": head + "def raw(self, text, context=None):\n    return text\n\n\nclass Walrus(Jot):\n    (put := raw)\n",
        "matched": head + ("def raw(self, text, context=None):\n    return text\n\n\nclass Matched(Jot):\n    match raw:\n"
                           "        case put:\n            pass\n"),
    }
    two_classes = ("import opti_oignon.jot as jm\nfrom opti_oignon.jot import get_jot\n\n\ndef choose():\n    return jm\n\n\n"
                   "class H:\n    def __init__(self):\n        self.store = get_jot()\n\n\nclass Other:\n"
                   "    def __init__(self):\n        self.store = choose()\n\n\n"
                   "def go(flag, x):\n    h = H() if flag else Other()\n    h.store.put(x)\n")
    one_class = two_classes.replace("h = H() if flag else Other()", "h = H()")
    reset_inside = carrier.replace("    def go(self, x):\n", "    def reset(self, jm_mod):\n"
                                   "        object.__setattr__(self, 'store', jm_mod)\n\n    def go(self, x):\n")
    dict_inside = carrier.replace("    def go(self, x):\n", "    def reset(self, jm_mod):\n"
                                  "        self.__dict__['store'] = jm_mod\n\n    def go(self, x):\n")
    guard, restore = _load()
    try:
        otherwise = guard.find_table_drift(_house_census(guard, {f"opti_oignon/{key}.py": text
                                                                 for key, text in bound_otherwise.items()}))
        either = guard.find_ungated(_house_census(guard, {"opti_oignon/h.py": two_classes}))
        only = guard.find_ungated(_house_census(guard, {"opti_oignon/h.py": one_class}))
        reset = guard.find_ungated(_house_census(guard, {"opti_oignon/h.py": reset_inside}))
        by_dict = guard.find_ungated(_house_census(guard, {"opti_oignon/h.py": dict_inside}))
    finally:
        restore()
    assert by_dict == [f"opti_oignon/h.py:{_line(dict_inside, 'self.store.put(')}: a jot write (put) in H.go() that "
                       f"none of its gates (own) covers"], by_dict
    assert sorted(otherwise) == sorted([f"gate own: opti_oignon/walrus.py: Walrus.put {overrides}",
                                        f"gate own: opti_oignon/matched.py: Matched.put {overrides}"]), otherwise
    assert either == [f"opti_oignon/h.py:{_line(two_classes, 'h.store.put(')}: a jot write (put) in go() that none of "
                      f"its gates (own) covers"], either
    assert only == [], only
    assert reset == [f"opti_oignon/h.py:{_line(reset_inside, 'self.store.put(')}: a jot write (put) in H.go() that "
                     f"none of its gates (own) covers"], reset


def test_sw44_a_route_is_a_routers_handler_and_its_closure_is_not():
    router = "from fastapi import APIRouter\n\nfrom opti_oignon.approve import caption\n\nrouter = APIRouter()\n\n\n"
    forms = {
        "closure": (router + "def run_agent(m, tools):\n    return tools\n\n\n@router.post('/chat')\ndef chat(m):\n"
                    "    def tool(a):\n        return caption(a, approve=True)\n    return run_agent(m, tools=[tool])\n"),
        "lambda": (router + "def run_agent(m, tools):\n    return tools\n\n\n@router.post('/chat')\ndef chat(m):\n"
                   "    return run_agent(m, tools=[lambda a: caption(a, approve=True)])\n"),
        "registry": ("from opti_oignon.approve import caption\n\n\nclass Tools:\n    def post(self, path):\n"
                     "        return lambda f: f\n\n\nTOOLS = Tools()\n\n\n@TOOLS.post('/caption')\ndef tool(x):\n"
                     "    return caption(x, approve=True)\n"),
        "direct": (router + "@router.post('/caption')\ndef route(x, approve: bool = False):\n"
                   "    return caption(x, approve=approve)\n"),
    }
    guard, restore = _load()
    try:
        _tables(guard, gates={"approved": guard.Gate("approval", _APPROVAL_HOME, "approve", (), "r")},
                gated={(_APPROVAL_HOME, "jot"): ("approved",)})
        found = {key: guard.find_ungated(_census(guard, {_APPROVAL_HOME: _APPROVAL, "opti_oignon/web.py": text}))
                 for key, text in forms.items()}
    finally:
        restore()
    authority = (f"{_APPROVAL_HOME}:{_line(_APPROVAL, 'put(')}: a jot write whose approval gate (approved) takes its "
                 f"authority from a caller that no route or named entry alone reaches")
    assert found == {"closure": [authority], "lambda": [authority], "registry": [authority], "direct": []}, found
    # Under another decorator, on a router whose attributes are replaced, or called by code, a handler carries no
    # gesture of the user's; its own body calling a method of its name does not call it.
    more = {
        "stacked": (router + "def keep(f):\n    return f\n\n\n@router.post('/caption')\n@keep\n"
                    "def route(x, approve: bool = False):\n    return caption(x, approve=approve)\n"),
        "rewired": (router + "class Tools:\n    def post(self, path):\n        return lambda f: f\n\n\n"
                    "router.post = Tools().post\n\n\n@router.post('/caption')\n"
                    "def route(x, approve: bool = False):\n    return caption(x, approve=approve)\n"),
        "called": (router + "@router.post('/caption')\ndef route(x, approve: bool = False):\n"
                   "    return caption(x, approve=approve)\n\n\ndef other(x):\n    return route(x, approve=True)\n"),
        "own_body": (router + "@router.post('/caption')\ndef route(x, store, approve: bool = False):\n"
                     "    store.route(x)\n    return caption(x, approve=approve)\n"),
    }
    # A closure an entry module defines is reached by its own callers: handed to an agent run, it is the model's.
    entry = {
        "closure": ("from opti_oignon.approve import caption\n\n\ndef agent(m, tools):\n    return tools\n\n\n"
                    "def run(m):\n    def tool(a):\n        return caption(a, approve=True)\n"
                    "    return agent(m, tools=[tool])\n"),
        "function": "from opti_oignon.approve import caption\n\n\ndef run(m):\n    return caption(m, approve=True)\n",
    }
    guard, restore = _load()
    try:
        _tables(guard, gates={"approved": guard.Gate("approval", _APPROVAL_HOME, "approve", (), "r")},
                gated={(_APPROVAL_HOME, "jot"): ("approved",)})
        more_found = {key: guard.find_ungated(_census(guard, {_APPROVAL_HOME: _APPROVAL, "opti_oignon/web.py": text}))
                      for key, text in more.items()}
        _tables(guard, gates={"approved": guard.Gate("approval", _APPROVAL_HOME, "approve", (), "r",
                                                     ("opti_oignon/cli.py",))},
                gated={(_APPROVAL_HOME, "jot"): ("approved",)})
        entry_found = {key: guard.find_ungated(_census(guard, {_APPROVAL_HOME: _APPROVAL, "opti_oignon/cli.py": text}))
                       for key, text in entry.items()}
    finally:
        restore()
    assert more_found == {"stacked": [authority], "rewired": [authority], "called": [authority], "own_body": []}, \
        more_found
    assert entry_found == {"closure": [authority], "function": []}, entry_found
    # A router whose attribute another module replaces may route anywhere.
    handler = router + "@router.post('/caption')\ndef route(x, approve: bool = False):\n    return caption(x, approve=approve)\n"
    hack = ("from opti_oignon.web import router\n\n\nclass Tools:\n    def post(self, path):\n        return lambda f: f\n\n\n"
            "router.post = Tools().post\n")
    guard, restore = _load()
    try:
        _tables(guard, gates={"approved": guard.Gate("approval", _APPROVAL_HOME, "approve", (), "r")},
                gated={(_APPROVAL_HOME, "jot"): ("approved",)})
        hacked = guard.find_ungated(_census(guard, {_APPROVAL_HOME: _APPROVAL, "opti_oignon/web.py": handler,
                                                    "opti_oignon/hack.py": hack}))
    finally:
        restore()
    assert hacked == [authority], hacked


def test_sw45_a_receiver_that_may_be_the_module_is_no_store_object():
    head = "import opti_oignon.a as a_mod\nfrom opti_oignon.jot import get_jot\n\n\n"
    forms = {
        "conditional": head + "def f(x, flag):\n    m = a_mod if flag else get_jot()\n    m.apply(x)\n",
        "container": head + "def f(x):\n    m = [a_mod, get_jot()][0]\n    m.apply(x)\n",
        "inline": head + "def f(x, flag):\n    (a_mod if flag else get_jot()).apply(x)\n",
        "chooser": (head + "def choose(flag):\n    return a_mod if flag else get_jot()\n\n\n"
                    "def f(x, flag):\n    choose(flag).apply(x)\n"),
        "table": head + "TABLE = {'k': a_mod, 'j': get_jot()}\n\n\ndef f(x):\n    TABLE.get('k').apply(x)\n",
        "held": (head + "class H:\n    def __init__(self):\n        self.m = (a_mod, get_jot())[0]\n\n"
                 "    def go(self, x):\n        self.m.apply(x)\n"),
        # A chooser held on ``self``, an accessor's name bound by a parameter's default or by a local definition.
        "chosen": (head + "def choose(flag):\n    return a_mod if flag else get_jot()\n\n\nclass H:\n"
                   "    def __init__(self, flag):\n        self.m = choose(flag)\n\n    def go(self, x):\n"
                   "        self.m.apply(x)\n"),
        "default": head + "def f(x, get_jot=lambda: a_mod):\n    get_jot().apply(x)\n",
        "local": head + "def f(x):\n    def get_jot():\n        return a_mod\n    get_jot().apply(x)\n",
    }
    controls = {
        "stored": head + "def f(x):\n    m = get_jot()\n    m.apply(x)\n",
        "accessor": head + "def f(x):\n    get_jot().apply(x)\n",
    }
    guard, restore = _load()
    try:
        _tables(guard, gates=_decided_gates(guard), gated={("opti_oignon/a.py", "jot"): ("keeper", "accept")})
        found = {key: _ungated_functions(guard, {"opti_oignon/a.py": _DECIDED, "opti_oignon/e.py": text})
                 for key, text in {**forms, **controls}.items()}
    finally:
        restore()
    assert found == {**{key: ["apply"] for key in forms}, **{key: [] for key in controls}}, found


def test_sw46_a_coroutine_or_a_generator_counts_only_where_it_is_consumed():
    head = "from opti_oignon.jot import get_jot\n\n\n"
    each = "def _each(xs):\n    for x in xs:\n        yield get_jot().drop(x)\n\n\n"
    later = "async def _later(x):\n    get_jot().drop(x)\n\n\n"
    forms = {
        "generator": head + each + "def accept(xs):\n    return _each(xs)\n",
        "coroutine": head + later + "def accept(x):\n    return _later(x)\n",
        "drained": head + each + "def accept(xs):\n    return list(_each(xs))\n",
        "iterated": head + each + "def accept(xs):\n    for _ in _each(xs):\n        pass\n",
        "awaited": head + later + "async def accept(x):\n    await _later(x)\n",
    }
    guard, restore = _load()
    try:
        _tables(guard, gates={"accept": guard.Gate("decision", "opti_oignon/a.py", "accept", (), "r")},
                gated={("opti_oignon/a.py", "jot"): ("accept",)})
        found = {key: _ungated_functions(guard, {"opti_oignon/a.py": text}) for key, text in forms.items()}
    finally:
        restore()
    assert found == {"generator": ["_each"], "coroutine": ["_later"], "drained": [], "iterated": [], "awaited": []}, \
        found
    # An async generator runs where it is iterated; a generator handed to a wrapper, or to a ``list`` the module
    # defines, is not drained on the spot; a partial of a coroutine runs only where its call is awaited.
    async_each = "async def _each(xs):\n    for x in xs:\n        yield get_jot().drop(x)\n\n\n"
    partial = "from functools import partial\n\nfrom opti_oignon.jot import get_jot\n\n\n"
    more = {
        "async_iterated": head + async_each + "async def accept(xs):\n    async for _ in _each(xs):\n        pass\n",
        "async_returned": head + async_each + "async def accept(xs):\n    return _each(xs)\n",
        "wrapped": head + each + "def accept(xs):\n    for _ in enumerate(_each(xs)):\n        pass\n",
        "masked": head + each + "def list(items):\n    return items\n\n\ndef accept(xs):\n    return list(_each(xs))\n",
        "partial_returned": partial + later + "def accept(x):\n    run = partial(_later, x)\n    return run()\n",
        "partial_awaited": partial + later + "async def accept(x):\n    run = partial(_later, x)\n    await run()\n",
    }
    guard, restore = _load()
    try:
        _tables(guard, gates={"accept": guard.Gate("decision", "opti_oignon/a.py", "accept", (), "r")},
                gated={("opti_oignon/a.py", "jot"): ("accept",)})
        more_found = {key: _ungated_functions(guard, {"opti_oignon/a.py": text}) for key, text in more.items()}
    finally:
        restore()
    assert more_found == {"async_iterated": [], "async_returned": ["_each"], "wrapped": ["_each"], "masked": ["_each"],
                          "partial_returned": ["_later"], "partial_awaited": []}, more_found


def test_sw47_a_header_runs_around_its_scope_and_a_lambdas_own_names_stand_in_front():
    verdict = "from opti_oignon.gate import allow, pick\nfrom opti_oignon.jot import get_jot, put_line\n\n\n"
    decided = {
        "default_in_decision": "from opti_oignon.jot import get_jot\n\n\ndef accept(x, done=get_jot().put('now')):\n"
                               "    return x\n",
        "decorator_in_decision": ("from opti_oignon.jot import get_jot\n\n\ndef keep(f):\n    return f\n\n\n"
                                  "@keep(get_jot().put)\ndef accept(x):\n    return x\n"),
        # The header of a function nested in the decision runs in the decision's own body: its own effect.
        "default_nested_in_decision": ("from opti_oignon.jot import get_jot\n\n\ndef accept(x):\n"
                                       "    def inner(done=get_jot().put(x)):\n        return done\n    return inner\n"),
    }
    checked = {
        "walrus_reset": (verdict + "def f(x):\n    ok = allow(x)\n\n    def inner(y=(ok := True)):\n        return y\n"
                         "    if not ok:\n        return None\n    get_jot().put(x)\n"),
        "lambda_comprehension": (verdict + "def f(messages, raw):\n    kept = pick(messages)\n"
                                 "    return lambda: [put_line(kept) for kept in raw]\n"),
        "lambda_walrus": (verdict + "def f(messages, raw):\n    kept = pick(messages)\n"
                          "    return lambda: (kept := raw) and put_line(kept)\n"),
        "closure_check": (verdict + "def f(x):\n    ok = allow(x)\n\n    def w():\n        if not ok:\n"
                          "            return None\n        get_jot().put(x)\n    w()\n"),
        "default_after_verdict": (verdict + "def f(x):\n    if not allow(x):\n        return None\n\n"
                                  "    def inner(y=get_jot().put(x)):\n        return y\n    return inner\n"),
        # A lambda's default runs where the lambda is built: after the verdict.
        "lambda_default_after_verdict": (verdict + "def f(x):\n    if not allow(x):\n        return None\n"
                                         "    g = lambda y=get_jot().put(x): y\n    return g\n"),
    }
    guard, restore = _load()
    try:
        gates = {"allow": guard.Gate("verdict", _GATE_HOME, "allow", (), "r"),
                 "pick": guard.Gate("filter", _GATE_HOME, "pick", (), "r")}
        gated = {(f"opti_oignon/{key}.py", "jot"): ("allow", "pick") for key in checked}
        for key in decided:
            gates[key] = guard.Gate("decision", f"opti_oignon/{key}.py", "accept", (), "r")
            gated[(f"opti_oignon/{key}.py", "jot")] = (key,)
        _tables(guard, gates=gates, gated=gated)
        found = guard.find_ungated(_census(guard, {
            _GATE_HOME: _GATE_TEXT, **{f"opti_oignon/{key}.py": text for key, text in {**decided, **checked}.items()}}))
    finally:
        restore()
    assert _modules_named(found) == sorted(["default_in_decision", "decorator_in_decision", "walrus_reset",
                                            "lambda_comprehension", "lambda_walrus"]), found
    # A parameter of a lambda or of a nested function, defaulted to raw data, stands in front of the filtered
    # name; and a write at module level is named so.
    refused = {
        "filter_lambda_default": (verdict + "def f(messages, raw):\n    kept = pick(messages)\n"
                                  "    return lambda kept=raw: put_line(kept)\n"),
        "filter_nested_default": (verdict + "def f(messages, raw):\n    kept = pick(messages)\n\n"
                                  "    def inner(kept=raw):\n        put_line(kept)\n    return inner\n"),
        "module_level": verdict + "get_jot().put('x')\n",
    }
    guard, restore = _load()
    try:
        _tables(guard, gates={"allow": guard.Gate("verdict", _GATE_HOME, "allow", (), "r"),
                              "pick": guard.Gate("filter", _GATE_HOME, "pick", (), "r")},
                gated={(f"opti_oignon/{key}.py", "jot"): ("allow", "pick") for key in refused})
        refused_found = guard.find_ungated(_census(guard, {
            _GATE_HOME: _GATE_TEXT, **{f"opti_oignon/{key}.py": text for key, text in refused.items()}}))
    finally:
        restore()
    assert _modules_named(refused_found) == sorted(refused), refused_found
    assert (f"opti_oignon/module_level.py:{_line(refused['module_level'], 'get_jot().put(')}: a jot write (put) at "
            f"module level that none of its gates (allow, pick) covers") in refused_found, refused_found


def test_sw48_a_star_import_may_rebind_any_name_the_module_reads():
    helpers = "def allow(x):\n    return True\n\n\ntempfile = None\n"
    gated = ("from opti_oignon.gate import allow\n{star}from opti_oignon.jot import get_jot\n\n\n"
             "def f(x):\n    if not allow(x):\n        return None\n    get_jot().put(x)\n")
    placed = ("import tempfile\n{star}from opti_oignon.jot import Jot\n\n\n"
              "def f():\n    Jot(path=tempfile.mkdtemp()).put('a')\n")
    star = "from opti_oignon.helpers import *\n"
    guard, restore = _load()
    try:
        _tables(guard, gates={"allow": guard.Gate("verdict", _GATE_HOME, "allow", (), "r")},
                gated={("opti_oignon/g_star.py", "jot"): ("allow",), ("opti_oignon/g_plain.py", "jot"): ("allow",)},
                exempt={("opti_oignon/p_star.py", "jot"): ("instance", "r"),
                        ("opti_oignon/p_plain.py", "jot"): ("instance", "r")})
        census = _census(guard, {_GATE_HOME: _GATE_TEXT, "opti_oignon/helpers.py": helpers,
                                 "opti_oignon/g_star.py": gated.format(star=star),
                                 "opti_oignon/g_plain.py": gated.format(star=""),
                                 "opti_oignon/p_star.py": placed.format(star=star),
                                 "opti_oignon/p_plain.py": placed.format(star="")})
        ungated = guard.find_ungated(census)
        failed = guard.find_failed_exemptions(census)
    finally:
        restore()
    assert _modules_named(ungated) == ["g_star"], ungated
    assert _modules_named(failed) == ["p_star"], failed
    # A star import, spaced or not, may bring in any function a module defines: a call of its name is a reference.
    star_callers = {
        "spaced": "from opti_oignon.a import *\n\n\ndef go(x):\n    apply(x)\n",
        "unspaced": "from opti_oignon.a import*\n\n\ndef go(x):\n    apply(x)\n",
        "none": "def go(x):\n    apply(x)\n",
    }
    # Under a star, ``runpy`` and ``hashlib`` are read the way that refuses more.
    runner = "import runpy\nfrom opti_oignon.helpers import *\n\n\ndef go():\n    runpy.run_module('opti_oignon.tool')\n"
    program = "from opti_oignon.jot import get_jot\n\nget_jot().put('a')\n"
    keyed = ("import hashlib\nfrom opti_oignon.helpers import *\nfrom opti_oignon.jot import get_jot\n\n\n"
             "def f():\n    get_jot().put('a', context=hashlib.sha256(b'x').hexdigest())\n")
    guard, restore = _load()
    try:
        _tables(guard, gates=_decided_gates(guard), gated={("opti_oignon/a.py", "jot"): ("keeper", "accept")})
        callers = {key: _ungated_functions(guard, {"opti_oignon/a.py": _DECIDED, "opti_oignon/e.py": text})
                   for key, text in star_callers.items()}
        _tables(guard, exempt={("opti_oignon/tool/__main__.py", "jot"): ("script", "r"),
                               ("opti_oignon/k_star.py", "jot"): ("keyed", "r")})
        star_failed = guard.find_failed_exemptions(_census(guard, {
            "opti_oignon/helpers.py": helpers, "opti_oignon/tool/__init__.py": "",
            "opti_oignon/tool/__main__.py": program, "opti_oignon/runner.py": runner, "opti_oignon/k_star.py": keyed}))
    finally:
        restore()
    assert callers == {"spaced": ["apply"], "unspaced": ["apply"], "none": []}, callers
    # A package that re-exports a module's functions by a star hands every caller of them through it.
    package = {"opti_oignon/pk/__init__.py": "from .core import *\n", "opti_oignon/pk/core.py": _DECIDED}
    caller = "from opti_oignon.pk import apply\n\n\ndef go(x):\n    apply(x)\n"
    guard, restore = _load()
    try:
        _tables(guard, gates={"keeper": guard.Gate("decision", "opti_oignon/pk/core.py", "Gatekeeper.write", (), "r"),
                              "accept": guard.Gate("decision", "opti_oignon/pk/core.py", "accept", (), "r")},
                gated={("opti_oignon/pk/core.py", "jot"): ("keeper", "accept")})
        through = guard.find_ungated(_census(guard, {**package, "opti_oignon/c.py": caller}))
        alone = guard.find_ungated(_census(guard, package))
    finally:
        restore()
    assert sorted(re.search(r" in (\w+)\(\)", line).group(1) for line in through) == ["apply"], through
    assert alone == [], alone
    assert sorted(star_failed) == sorted([
        f"opti_oignon/tool/__main__.py:{_line(program, '.put(')}: a jot write outside the module's __main__ block",
        f"opti_oignon/k_star.py:{_line(keyed, '.put(')}: a jot write that binds no context key (context)"]), star_failed


def test_sw49_a_key_is_live_only_when_the_function_that_writes_computes_it():
    head = "import hashlib\n\nfrom opti_oignon.jot import get_jot\n\n"
    constant = {
        "digest": head + "NOCTX = hashlib.sha256(b'no-context').hexdigest()\n\n\n"
                         "def f():\n    get_jot().put('a', context=NOCTX)\n",
        "upper": head + "\ndef f():\n    get_jot().put('a', context='same'.upper())\n",
        "class_constant": head + "\nclass K:\n    CTX = 'k'\n\n\ndef f():\n    get_jot().put('a', context=K.CTX)\n",
        "subscript": head + "CTXS = ['a']\n\n\ndef f():\n    get_jot().put('a', context=CTXS[0])\n",
        "augmented": head + "\ndef f():\n    ctx = 'a'\n    ctx += '!'\n    get_jot().put('a', context=ctx)\n",
        "default": head + "\ndef f(ctx='same'):\n    get_jot().put('a', context=ctx)\n",
        "shadow": (head + "ctx = 'same'\n\n\ndef g(ctx):\n    return ctx\n\n\n"
                          "def f():\n    get_jot().put('a', context=ctx)\n"),
        "unguarded": (head + "\ndef f(fp):\n    key = ''\n    if fp:\n        key = compute(fp)\n"
                             "    get_jot().put('a', context=key)\n"),
        # A constant on one path: a side of an ``or``, a branch of a conditional, a walrus, a loop over constants.
        "or_constant": head + "NOCTX = 'none'\n\n\ndef f(fp):\n    get_jot().put('a', context=fp or NOCTX)\n",
        "conditional": head + "\ndef f(fp):\n    get_jot().put('a', context=fp if fp else 'none')\n",
        "walrus": head + "\ndef f():\n    get_jot().put('a', context=(k := 'same'))\n    return k\n",
        "loop_constants": head + "\ndef f():\n    for key in ('a', 'b'):\n        get_jot().put('a', context=key)\n",
        # A sentinel re-armed after its test, in a nested block or a loop, reaches the write; one bound in a loop
        # around the test is not read.
        "rearmed_nested": (head + "\ndef f(fp):\n    key = ''\n    if fp:\n        key = compute(fp)\n    if key:\n"
                                  "        if fp == 'x':\n            key = ''\n        get_jot().put('a', context=key)\n"),
        "rearmed_loop": (head + "\ndef f(fp, items):\n    key = ''\n    if fp:\n        key = compute(fp)\n    if key:\n"
                                "        for item in items:\n            get_jot().put(item, context=key)\n"
                                "            key = ''\n"),
        "sentinel_in_loop": (head + "\ndef f(fps):\n    for fp in fps:\n        key = ''\n        if fp:\n"
                                    "            key = compute(fp)\n        if key:\n"
                                    "            get_jot().put('a', context=key)\n"),
        # A sentinel an augmentation by a constant carries on; a key a nested scope resets; a constant reached
        # through ``with``, a constant container's ``get``, ``range``, a function of the module or a lambda
        # handed constants, ``json.dumps`` of constants.
        "aug_sentinel": (head + "\ndef f(fp):\n    key = ''\n    key += '-v2'\n    if key:\n"
                                "        get_jot().put('a', context=key)\n"),
        "nonlocal_reset": (head + "\ndef f(fp):\n    key = compute(fp)\n\n    def reset():\n        nonlocal key\n"
                                  "        key = ''\n    reset()\n    get_jot().put('a', context=key)\n"),
        "with_constant": (head + "from contextlib import nullcontext\n\n\ndef f():\n    with nullcontext('k') as key:\n"
                                 "        get_jot().put('a', context=key)\n"),
        "get_constant": head + "KD = {'a': 'k'}\n\n\ndef f():\n    get_jot().put('a', context=KD.get('a'))\n",
        "range_key": head + "\ndef f():\n    for key in range(3):\n        get_jot().put('a', context=key)\n",
        "local_function": (head + "\ndef _kf():\n    return 'k'\n\n\ndef f():\n"
                                  "    get_jot().put('a', context=_kf())\n"),
        "lambda_key": head + "_kl = lambda: 'k'\n\n\ndef f():\n    get_jot().put('a', context=_kl())\n",
        "lambda_inline": head + "\ndef f():\n    get_jot().put('a', context=(lambda: 'k')())\n",
        "dumps": head + "import json\n\n\ndef f():\n    get_jot().put('a', context=json.dumps({'a': 1}))\n",
        "with_sentinel": (head + "from contextlib import nullcontext\n\n\ndef f():\n    key = ''\n"
                                 "    with nullcontext('k') as key:\n        pass\n    if key:\n"
                                 "        get_jot().put('a', context=key)\n"),
    }
    live = {
        "parameter": head + "\ndef f(ctx):\n    get_jot().put('a', context=ctx)\n",
        "computed": head + "\ndef f(messages):\n    fp = fingerprint(messages)\n    get_jot().put('a', context=fp)\n",
        "guarded": (head + "\ndef f(fp):\n    key = ''\n    if fp:\n        key = compute(fp)\n    if key:\n"
                           "        get_jot().put('a', context=key)\n"),
        # An augmented assignment brings no constant of its own: a live key stays live.
        "augmented_live": (head + "\ndef f(messages):\n    key = fingerprint(messages)\n    key += '-v2'\n"
                                  "    get_jot().put('a', context=key)\n"),
        # A function of the module handed what the writer was handed is live.
        "live_function": (head + "\ndef _kf(m):\n    return m\n\n\ndef f(messages):\n"
                                 "    get_jot().put('a', context=_kf(messages))\n"),
    }
    forms = {**constant, **live}
    guard, restore = _load()
    try:
        _tables(guard, exempt={(f"opti_oignon/{key}.py", "jot"): ("keyed", "r") for key in forms})
        failed = guard.find_failed_exemptions(_census(guard, {f"opti_oignon/{key}.py": t for key, t in forms.items()}))
    finally:
        restore()
    assert _modules_named(failed) == sorted(constant), failed


def test_sw50_a_module_is_read_by_its_syntax_tree_never_its_text():
    forms = {
        "joined": "from.jot import get_jot\n\n\ndef f():\n    get_jot().put('x')\n",
        "dotless": "from .import jot\n\n\ndef f():\n    jot.get_jot().put('x')\n",
        "spaced": "from opti_oignon . jot import get_jot\n\n\ndef f():\n    get_jot().put('x')\n",
        "commented": "from opti_oignon import (  # the store\n    jot,\n)\n\n\ndef f():\n    jot.get_jot().put('x')\n",
        "parenthesized": "from opti_oignon import pkg\n\n\ndef f():\n    (pkg).inner.get_jot().put('x')\n",
        "split": "from opti_oignon import pkg\n\n\ndef f():\n    (pkg\n     .inner\n     .get_jot()\n     .put('x'))\n",
        "rebound": "from opti_oignon import pkg\n\nP = pkg\n\n\ndef f():\n    P.inner.get_jot().put('x')\n",
        "top": ("import importlib\n\n\ndef f():\n    m = importlib.import_module('opti_oignon')\n"
                "    m.jot.get_jot().put('x')\n"),
        "concatenated": ("import importlib\n\n\ndef f():\n    m = importlib.import_module('opti_oignon.' 'jot')\n"
                         "    m.get_jot().put('x')\n"),
    }
    shared = {"opti_oignon/pkg/__init__.py": "", "opti_oignon/pkg/inner.py": "from opti_oignon.jot import get_jot\n"}
    guard, restore = _load()
    try:
        _tables(guard)
        sites = _sites(_census(guard, {**shared, **{f"opti_oignon/{key}.py": text for key, text in forms.items()}}))
    finally:
        restore()
    assert {key: sites.get(f"opti_oignon/{key}.py") for key in forms} == {key: [("jot", "put", "f")] for key in forms}, \
        sites


def test_sw51_a_proof_is_read_as_pytest_reads_the_selection(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    home = "import opti_oignon.gate\n"
    suites = {
        "test_pre_contracts.py": home + "\n\ndef test_p1_x():\n    pass\n\n\ndef test_p10_y():\n    pass\n",
        "test_dot_contracts.py": home + "\n\ndef test_d1_x():\n    pass\n",
        "test_inherit_contracts.py": home + "\n\nclass Base:\n    pass\n\n\nclass TestKid(Base):\n"
                                           "    def test_i1_x(self):\n        pass\n",
        "test_del_contracts.py": home + "\n\ndef test_e1_x():\n    pass\n\n\ndel test_e1_x\n",
        "test_hidden_contracts.py": home + "\nif False:\n    def test_g1_x():\n        pass\n",
        "test_alias_contracts.py": home + "import pytest\n\n_off = pytest.mark.skip\n\n\n@_off\ndef test_a1_x():\n    pass\n",
        "test_raise_contracts.py": home + "import unittest\n\nraise unittest.SkipTest('later')\n\n\n"
                                         "def test_r1_x():\n    pass\n",
        "test_ann_contracts.py": home + "\n__test__: bool = False\n\n\ndef test_t1_x():\n    pass\n",
        "test_async_contracts.py": home + "\n\nasync def test_s1_x():\n    pass\n",
        "test_plain_contracts.py": home + "\n\ndef test_c1_x():\n    pass\n",
    }
    for name, text in suites.items():
        (tests / name).write_text(text, encoding="utf-8")
    rule = ('[tool.pytest.ini_options]\naddopts = """\n    --deselect=tests/test_pre_contracts.py::test_p1\n'
            '    --ignore=./tests/test_dot_contracts.py\n"""\n')
    (tmp_path / "pyproject.toml").write_text(rule, encoding="utf-8")
    contracts = {"prefix": "p10", "dot": "d1", "inherit": "i1", "deleted": "e1", "hidden": "g1", "aliased": "a1",
                 "raised": "r1", "annotated": "t1", "asynchronous": "s1", "plain": "c1"}
    guard, restore = _load()
    try:
        _tables(guard, gates={name: guard.Gate("verdict", _GATE_HOME, "allow", (contract,), "r")
                              for name, contract in contracts.items()})
        proofs = guard.find_gate_proofs_missing(_census(guard, {_GATE_HOME: _GATE_TEXT}, root=tmp_path))
        (tests / "conftest.py").write_text("def pytest_collection_modifyitems(items):\n    pass\n", encoding="utf-8")
        hooked = guard.find_gate_proofs_missing(_census(guard, {_GATE_HOME: _GATE_TEXT}, root=tmp_path))
        (tests / "conftest.py").unlink()
        (tmp_path / "pyproject.toml").write_text(rule.replace('"""\n', '"""\n    -vk real\n', 1), encoding="utf-8")
        combined = guard.find_gate_proofs_missing(_census(guard, {_GATE_HOME: _GATE_TEXT}, root=tmp_path))
    finally:
        restore()
    assert sorted(proofs) == sorted(f"gate {gate}: contract {contract} names no test function that runs"
                                    for gate, contract in contracts.items() if gate != "plain"), proofs
    unread = "the selection rule holds what the census cannot read ({}): no gate proof can be read under it"
    assert sorted(set(hooked) - set(proofs)) == [unread.format("tests/conftest.py: pytest_collection_modifyitems")], \
        hooked
    assert sorted(set(combined) - set(proofs)) == [unread.format("-vk real")], combined
    # A skip under another name, a suite that generates its tests, runs its own function at load or rebinds a
    # test by ``global``, a deselect in quotes: no proof. A deselect pytest reads letter for letter (``./``) removes
    # nothing. A pytest.ini, ``collect_ignore`` and ``pytest_plugins`` are what the census cannot read.
    more = tmp_path / "more"
    (more / "tests").mkdir(parents=True)
    more_suites = {
        "test_alias_skip_contracts.py": home + "from pytest import skip as later\n\n\ndef test_k1_x():\n    later('no')\n",
        "test_generate_contracts.py": home + "\n\ndef pytest_generate_tests(metafunc):\n    pass\n\n\n"
                                             "def test_n1_x():\n    pass\n",
        "test_load_contracts.py": home + "\n\ndef _boom():\n    raise RuntimeError\n\n\n_boom()\n\n\n"
                                         "def test_l1_x():\n    pass\n",
        "test_global_contracts.py": home + "\n\ndef test_m1_x():\n    pass\n\n\ndef rebind():\n"
                                           "    global test_m1_x\n    test_m1_x = None\n",
        "test_quoted_contracts.py": home + "\n\ndef test_q1_x():\n    pass\n",
        "test_literal_contracts.py": home + "\n\ndef test_v1_x():\n    pass\n",
        # A ``fail`` is loud, and a common word: no silent skip.
        "test_loud_contracts.py": home + "import pytest\n\n\ndef test_f1_x():\n    if False:\n        pytest.fail('never')\n",
        # An exit at load or in a test may end the session green, its tests unrun; one under ``__main__`` runs
        # only as a program.
        "test_exit_contracts.py": home + "import pytest\n\npytest.exit('done', returncode=0)\n\n\ndef test_x1_x():\n    pass\n",
        "test_quit_contracts.py": home + "import sys\n\n\ndef test_y1_x():\n    sys.exit(0)\n",
        "test_main_contracts.py": (home + "import sys\n\n\ndef test_z1_x():\n    pass\n\n\nif __name__ == '__main__':\n"
                                          "    sys.exit(0)\n"),
    }
    for name, text in more_suites.items():
        (more / "tests" / name).write_text(text, encoding="utf-8")
    (more / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = "--deselect \'tests/test_quoted_contracts.py::test_q1\' '
        '--deselect=./tests/test_literal_contracts.py::test_v1"\n', encoding="utf-8")
    more_contracts = {"alias_skip": "k1", "generated": "n1", "loaded": "l1", "rebound": "m1", "quoted": "q1",
                      "literal": "v1", "loud": "f1", "exit_load": "x1", "exit_test": "y1", "exit_main": "z1"}
    sources = {}
    guard, restore = _load()
    try:
        _tables(guard, gates={name: guard.Gate("verdict", _GATE_HOME, "allow", (contract,), "r")
                              for name, contract in more_contracts.items()})
        more_proofs = guard.find_gate_proofs_missing(_census(guard, {_GATE_HOME: _GATE_TEXT}, root=more))
        for name, text in (("pytest.ini", "[pytest]\n"), ("tests/conftest.py", "collect_ignore = ['x']\n"),
                           ("tests/conftest.py", "pytest_plugins = ['x']\n"), ("tests/pytest.ini", "[pytest]\n")):
            (more / name).write_text(text, encoding="utf-8")
            sources[f"{name}: {text}"] = sorted(set(guard.find_gate_proofs_missing(
                _census(guard, {_GATE_HOME: _GATE_TEXT}, root=more))) - set(more_proofs))
            (more / name).unlink()
    finally:
        restore()
    assert sorted(more_proofs) == sorted(f"gate {gate}: contract {contract} names no test function that runs"
                                         for gate, contract in more_contracts.items()
                                         if gate not in ("literal", "loud", "exit_main")), more_proofs
    assert sources == {"pytest.ini: [pytest]\n": [unread.format("pytest.ini")],
                       "tests/conftest.py: collect_ignore = ['x']\n": [
                           unread.format("tests/conftest.py: collect_ignore")],
                       "tests/conftest.py: pytest_plugins = ['x']\n": [
                           unread.format("tests/conftest.py: pytest_plugins")],
                       "tests/pytest.ini: [pytest]\n": [unread.format("tests/pytest.ini")]}, sources


def test_sw52_a_name_names_every_definition_and_every_object_it_may():
    helpers = {
        "opti_oignon/helper_a.py": "def write_it(store, x):\n    store.put(x)\n",
        "opti_oignon/helper_b.py": "def write_it(handed, x):\n    handed.drop(x)\n",
        "opti_oignon/helper_c.py": "def keep(kept, x):\n    kept.put(x)\n",
        "opti_oignon/helper_d.py": "",
    }
    branches = ("from opti_oignon.jot import get_jot\n\ntry:\n    from opti_oignon.helper_a import write_it\n"
                "except ImportError:\n    from opti_oignon.helper_b import write_it\n\n\n"
                "def f():\n    write_it(get_jot(), 'x')\n")
    fallback = ("from opti_oignon.jot import get_jot\n\ntry:\n    from opti_oignon.helper_c import keep\n"
                "except ImportError:\n    def keep(local, x):\n        return None\n\n\n"
                "def f():\n    keep(get_jot(), 'x')\n")
    carrier = ("import opti_oignon.helper_d as holder\nfrom opti_oignon.jot import get_jot\n\n\nclass Holder:\n"
               "    def __init__(self):\n        self.store = get_jot()\n\n\ndef make_holder():\n    return Holder()\n\n\n"
               "def f():\n    holder = make_holder()\n    holder.store.put('x')\n")
    guard, restore = _load()
    try:
        _tables(guard)
        sites = _sites(_census(guard, {**helpers, "opti_oignon/branches.py": branches,
                                       "opti_oignon/fallback.py": fallback, "opti_oignon/carrier.py": carrier}))
    finally:
        restore()
    assert {rel: sites.get(rel) for rel in ("opti_oignon/helper_a.py", "opti_oignon/helper_b.py",
                                            "opti_oignon/helper_c.py", "opti_oignon/carrier.py")} == {
        "opti_oignon/helper_a.py": [("jot", "put", "write_it")],
        "opti_oignon/helper_b.py": [("jot", "drop", "write_it")],
        "opti_oignon/helper_c.py": [("jot", "put", "keep")],
        "opti_oignon/carrier.py": [("jot", "put", "f")]}, sites
    # Re-exports that branch at every step reach every definition, each place visited once: four modules a
    # level, nine levels, each module re-exporting from all four of the next -- four to the eighth paths.
    letters, levels = "abcd", 9
    diamond = {f"opti_oignon/dm{i}{x}.py": "".join(f"from opti_oignon.dm{i + 1}{y} import w\n" for y in letters)
               for i in range(levels - 1) for x in letters}
    diamond.update({f"opti_oignon/dm{levels - 1}{x}.py": "def w(store, x):\n    store.put(x)\n" for x in letters})
    user = "from opti_oignon.dm0a import w\nfrom opti_oignon.jot import get_jot\n\n\ndef f():\n    w(get_jot(), 'x')\n"
    guard, restore = _load()
    try:
        _tables(guard)
        visits = []
        imports = guard.Package.imports
        guard.Package.imports = lambda package, rel: visits.append(rel) or imports(package, rel)
        deep = _sites(_census(guard, {**diamond, "opti_oignon/user.py": user}))
    finally:
        restore()
    assert {rel: deep.get(rel) for rel in diamond if rel.startswith(f"opti_oignon/dm{levels - 1}")} == {
        f"opti_oignon/dm{levels - 1}{x}.py": [("jot", "put", "w")] for x in letters}, deep
    # Fewer reads of the modules' imports than a quarter of the paths: each place is visited once.
    assert len(visits) < len(letters) ** (levels - 1) // 4, len(visits)


def test_sw53_a_class_table_and_an_inherited_method_in_a_table_are_followed():
    helper = "class Base:\n    def _drop(self, dropped, x):\n        dropped.drop(x)\n"
    boxed = ("from opti_oignon.jot import get_jot\n\n\ndef _add(added, x):\n    added.put(x)\n\n\nclass Box:\n"
             "    TABLE = {'add': _add}\n\n    def handle(self, x):\n        return self.TABLE['add'](get_jot(), x)\n")
    kid = ("from opti_oignon.helper import Base\nfrom opti_oignon.jot import get_jot\n\n\nclass Kid(Base):\n"
           "    def handle(self, x):\n        table = {'drop': self._drop}\n        return table['drop'](get_jot(), x)\n")
    guard, restore = _load()
    try:
        _tables(guard)
        sites = _sites(_census(guard, {"opti_oignon/helper.py": helper, "opti_oignon/boxed.py": boxed,
                                       "opti_oignon/kid.py": kid}))
    finally:
        restore()
    assert sites.get("opti_oignon/boxed.py") == [("jot", "put", "_add")], sites
    assert sites.get("opti_oignon/helper.py") == [("jot", "drop", "Base._drop")], sites


def test_sw54_rewiring_a_store_object_outside_its_house_is_a_site_and_unmakes_an_own_object():
    rewired = ("from opti_oignon.jot import get_jot\n\n\ndef f(user_path):\n    j = get_jot()\n    j._path = user_path\n"
               "    j.path = user_path\n    setattr(j, 'path', user_path)\n    j.__init__(user_path)\n")
    moved = ("import tempfile\n\nfrom opti_oignon.jot import Jot\n\n\ndef f(user_path):\n"
             "    j = Jot(path=tempfile.mkdtemp())\n    j._path = user_path\n    j.put('a')\n")
    guard, restore = _load()
    try:
        _tables(guard, exempt={("opti_oignon/moved.py", "jot"): ("instance", "r")})
        census = _census(guard, {"opti_oignon/rewired.py": rewired, "opti_oignon/moved.py": moved})
        sites = _sites(census)
        failed = guard.find_failed_exemptions(census)
    finally:
        restore()
    assert sites.get("opti_oignon/rewired.py") == sorted([("jot", "_path", "f"), ("jot", "path", "f"),
                                                           ("jot", "path", "f"), ("jot", "__init__", "f")]), sites
    assert sorted(failed) == sorted(
        f"opti_oignon/moved.py:{_line(moved, needle)}: a jot write on an object the module did not build itself at "
        f"a temporary place" for needle in ("j._path", "j.put(")), failed
    # Rebuilt through its class or through ``object``, a store object is rewired too, and is no own object.
    rebuilt = ("from opti_oignon.jot import Jot, get_jot\n\n\ndef f(user_path):\n    j = get_jot()\n"
               "    Jot.__init__(j, user_path)\n    object.__setattr__(j, 'path', user_path)\n")
    unmade = {
        "reinit": ("import tempfile\n\nfrom opti_oignon.jot import Jot\n\n\ndef f(user_path):\n"
                   "    j = Jot(path=tempfile.mkdtemp())\n    Jot.__init__(j, user_path)\n    j.put('a')\n"),
        "reset": ("import tempfile\n\nfrom opti_oignon.jot import Jot\n\n\ndef f(user_path):\n"
                  "    j = Jot(path=tempfile.mkdtemp())\n    object.__setattr__(j, 'path', user_path)\n    j.put('a')\n"),
    }
    guard, restore = _load()
    try:
        _tables(guard, exempt={(f"opti_oignon/{key}.py", "jot"): ("instance", "r") for key in unmade})
        census = _census(guard, {"opti_oignon/rebuilt.py": rebuilt,
                                 **{f"opti_oignon/{key}.py": text for key, text in unmade.items()}})
        rebuilt_sites = _sites(census).get("opti_oignon/rebuilt.py")
        unmade_failed = guard.find_failed_exemptions(census)
    finally:
        restore()
    assert rebuilt_sites == [("jot", "__init__", "f"), ("jot", "__setattr__", "f")], rebuilt_sites
    needles = {"reinit": ("Jot.__init__(", "j.put("), "reset": ("object.__setattr__(", "j.put(")}
    assert sorted(unmade_failed) == sorted(
        f"opti_oignon/{key}.py:{_line(unmade[key], needle)}: a jot write on an object the module did not build "
        f"itself at a temporary place" for key in unmade for needle in needles[key]), unmade_failed


def test_sw55_a_package_attribute_a_starred_fromlist_a_lookup_and_a_getattr_are_followed():
    forms = {
        "getattr_pkg": "from opti_oignon import pkg\n\n\ndef f():\n    getattr(pkg, 'inner').get_jot().put('x')\n",
        "starred": ("NAMES = ['put_line']\n\n\ndef f():\n"
                    "    m = __import__('opti_oignon.jot', globals(), locals(), [*NAMES])\n    m.put_line('x')\n"),
        "aliased_pkg": "from opti_oignon import pkg\n\n\ndef f():\n    pkg.alias_inner.get_jot().put('x')\n",
        "base_getattr": ("import opti_oignon.jot as jm\n\n\nclass Mine(getattr(jm, 'Jot')):\n    def save(self, x):\n"
                         "        self.put(x)\n"),
        "getattr_sink": "import opti_oignon.jot as jm\n\n\ndef f():\n    getattr(jm, 'put_line')('x')\n",
        "looked_up": ("import importlib\n\nfrom opti_oignon.jot import get_jot\n\n\ndef f(obj):\n"
                      "    importlib.import_module('opti_oignon.helper')\n    obj.write(get_jot(), 'x')\n"),
    }
    shared = {"opti_oignon/pkg/__init__.py": "from . import inner as alias_inner\n",
              "opti_oignon/pkg/inner.py": "from opti_oignon.jot import get_jot\n",
              "opti_oignon/helper.py": "class Other:\n    def write(self, written, x):\n        written.put(x)\n"}
    guard, restore = _load()
    try:
        _tables(guard)
        sites = _sites(_census(guard, {**shared, **{f"opti_oignon/{key}.py": text for key, text in forms.items()}}))
    finally:
        restore()
    put = [("jot", "put", "f")]
    assert {key: sites.get(f"opti_oignon/{key}.py") for key in forms} == {
        "getattr_pkg": put, "starred": [("jot", "put_line", "f")], "aliased_pkg": put,
        "base_getattr": [("jot", "put", "Mine.save")], "getattr_sink": [("jot", "put_line", "f")],
        "looked_up": None}, sites
    assert sites.get("opti_oignon/helper.py") == [("jot", "put", "Other.write")], sites
    # ``__import__`` with a starred fromlist may hand back the top package, whose attribute is the store module; a
    # plain module relays a store module under another name, by attribute or by import; a package re-exports a
    # module from outside its folder; a default handed back by ``getattr``, ``get`` or ``next`` is followed.
    more = {
        "head_package": ("NAMES = ['x']\n\n\ndef f():\n"
                         "    m = __import__('opti_oignon.pkg.inner', globals(), locals(), [*NAMES])\n"
                         "    m.jot.get_jot().put('x')\n"),
        "relayed": "from opti_oignon import relay\n\n\ndef f():\n    relay.storage.get_jot().put('x')\n",
        "relayed_name": "from opti_oignon.relay import storage\n\n\ndef f():\n    storage.get_jot().put('x')\n",
        "foreign_reexport": "from opti_oignon import pkg2\n\n\ndef f():\n    pkg2.store.get_jot().put('x')\n",
        "defaults": ("from opti_oignon.jot import get_jot\n\n\ndef f(obj, table):\n"
                     "    getattr(obj, 'store', get_jot()).put('x')\n    table.get('k', get_jot()).put('y')\n"
                     "    next(iter(table), get_jot()).put('z')\n"),
    }
    more_shared = {"opti_oignon/relay.py": "import opti_oignon.jot as storage\n",
                   "opti_oignon/pkg2/__init__.py": "from opti_oignon import jot as store\n"}
    guard, restore = _load()
    try:
        _tables(guard)
        more_sites = _sites(_census(guard, {**shared, **more_shared,
                                            **{f"opti_oignon/{key}.py": text for key, text in more.items()}}))
    finally:
        restore()
    assert {key: more_sites.get(f"opti_oignon/{key}.py") for key in more} == {
        **{key: put for key in more if key != "defaults"}, "defaults": put * 3}, more_sites
    # A plain module relays a store module by ``from ... import ... as``, or by a lookup by name at its module
    # level; a package binds a name to a module in both branches of a ``try``: every one is followed.
    relays = {
        "relay_from": "from opti_oignon import relay2\n\n\ndef f():\n    relay2.storage.put_line('x')\n",
        "relay_lookup": "from opti_oignon import relay3\n\n\ndef f():\n    relay3.storage.get_jot().put('x')\n",
        "try_branches": "from opti_oignon.pkg_try import backend\n\n\ndef f():\n    backend.get_jot().put('x')\n",
    }
    relayed = {"opti_oignon/relay2.py": "from opti_oignon import jot as storage\n",
               "opti_oignon/relay3.py": "import importlib\n\nstorage = importlib.import_module('opti_oignon.jot')\n",
               "opti_oignon/pkg_try/__init__.py": ("try:\n    from opti_oignon import jot as backend\nexcept ImportError:\n"
                                                   "    from opti_oignon import other as backend\n"),
               "opti_oignon/other.py": ""}
    guard, restore = _load()
    try:
        _tables(guard)
        relay_sites = _sites(_census(guard, {**relayed, **{f"opti_oignon/{key}.py": text for key, text in relays.items()}}))
    finally:
        restore()
    assert {key: relay_sites.get(f"opti_oignon/{key}.py") for key in relays} == {
        "relay_from": [("jot", "put_line", "f")], "relay_lookup": put, "try_branches": put}, relay_sites


def test_sw56_a_gate_defined_twice_answers_for_nothing_and_its_imports_answer_for_it():
    lookalike = _GATE_TEXT.replace("def allow(x):", "from opti_oignon.jot import get_jot\n\n\ndef allow(x):", 1) + (
        "\n\nclass Fake:\n    def allow(self, x):\n        return True\n\n    def save(self, x):\n"
        "        if not self.allow(x):\n            return None\n        get_jot().put(x)\n")
    twice = "from opti_oignon.jot import get_jot\n\n\ndef accept(x):\n    get_jot().put(x)\n\n\ndef accept(x):\n    return None\n"
    imported_twice = ("from opti_oignon.gate import allow\nfrom opti_oignon.jot import get_jot\n\n\ndef f(x):\n"
                      "    from opti_oignon.gate import allow\n    if not allow(x):\n        return None\n    get_jot().put(x)\n")
    reexported = ("from opti_oignon.pkg2 import allow\nfrom opti_oignon.jot import get_jot\n\n\ndef f(x):\n"
                  "    if not allow(x):\n        return None\n    get_jot().put(x)\n")
    annotated = ("from functools import partial\nfrom typing import Callable\n\nfrom opti_oignon.jot import get_jot\n\n\n"
                 "def _private(x):\n    get_jot().drop(x)\n\n\ndef accept(x):\n    run: Callable = partial(_private, x)\n"
                 "    return run()\n")
    hooked = _DECIDED.replace("\n\ndef apply(x):", "\n\nclass HOOKS:\n    @staticmethod\n    def override(f):\n"
                              "        return f\n\n\n@HOOKS.override\ndef apply(x):")
    guard, restore = _load()
    try:
        gates = {"allow": guard.Gate("verdict", _GATE_HOME, "allow", (), "r"),
                 "twice": guard.Gate("decision", "opti_oignon/twice.py", "accept", (), "r"),
                 "annotated": guard.Gate("decision", "opti_oignon/annotated.py", "accept", (), "r")}
        _tables(guard, gates=gates, gated={(_GATE_HOME, "jot"): ("allow",), ("opti_oignon/twice.py", "jot"): ("twice",),
                                            ("opti_oignon/imported_twice.py", "jot"): ("allow",),
                                            ("opti_oignon/reexported.py", "jot"): ("allow",),
                                            ("opti_oignon/annotated.py", "jot"): ("annotated",)})
        others = {"opti_oignon/pkg2/__init__.py": "from opti_oignon.gate import allow\n",
                  "opti_oignon/twice.py": twice, "opti_oignon/imported_twice.py": imported_twice,
                  "opti_oignon/reexported.py": reexported, "opti_oignon/annotated.py": annotated}
        found = guard.find_ungated(_census(guard, {_GATE_HOME: _GATE_TEXT.replace(
            "def allow(x):", "from opti_oignon.jot import get_jot\n\n\ndef allow(x):", 1), **others}))
        # A home that defines a lookalike of the gate besides it answers for no gate, in any module.
        lookalike_found = guard.find_ungated(_census(guard, {_GATE_HOME: lookalike, **others}))
        _tables(guard, gates=_decided_gates(guard), gated={("opti_oignon/a.py", "jot"): ("keeper", "accept")})
        hook = _ungated_functions(guard, {"opti_oignon/a.py": hooked})
    finally:
        restore()
    assert _modules_named(found) == ["twice"], found
    assert _modules_named(lookalike_found) == ["gate", "imported_twice", "reexported", "twice"], lookalike_found
    assert hook == ["apply"], hook
    # A decision bound again, a gate bound again at home or defined only for a type checker, and a re-exporter
    # that rebinds the gate's name answer for nothing; a type checker's stub beside the definition changes nothing.
    rebound = ("from opti_oignon.jot import get_jot\n\n\ndef keep(f):\n    return f\n\n\ndef accept(x):\n"
               "    get_jot().put(x)\n\n\naccept = keep(accept)\n")
    relinked = ("from opti_oignon.pkg3 import allow\nfrom opti_oignon.jot import get_jot\n\n\ndef f(x):\n"
                "    if not allow(x):\n        return None\n    get_jot().put(x)\n")
    homes = {
        "plain": _GATE_TEXT,
        "reassigned": _GATE_TEXT + "\n\nallow = bool\n",
        "stub_only": ("from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    def allow(x):\n"
                      "        return bool(x)\n\n\n" + _GATE_TEXT.split("\n\n\n", 1)[1]),
        "stub_beside": (_GATE_TEXT + "\n\nfrom typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    def allow(x):\n"
                                     "        return True\n"),
        # A function that declares the gate global, and a walrus at home, may bind it anew.
        "global_rebind": _GATE_TEXT + "\n\ndef relax():\n    global allow\n    allow = bool\n",
        "walrus_home": _GATE_TEXT + "\n\nif (allow := bool):\n    pass\n",
    }
    users = {"opti_oignon/pkg2/__init__.py": "from opti_oignon.gate import allow\n",
             "opti_oignon/pkg3/__init__.py": "from opti_oignon.gate import allow\n\nallow = bool\n",
             "opti_oignon/imported_twice.py": imported_twice, "opti_oignon/reexported.py": reexported,
             "opti_oignon/relinked.py": relinked, "opti_oignon/rebound.py": rebound}
    guard, restore = _load()
    try:
        _tables(guard, gates={"allow": guard.Gate("verdict", _GATE_HOME, "allow", (), "r"),
                              "rebound": guard.Gate("decision", "opti_oignon/rebound.py", "accept", (), "r")},
                gated={("opti_oignon/imported_twice.py", "jot"): ("allow",),
                       ("opti_oignon/reexported.py", "jot"): ("allow",), ("opti_oignon/relinked.py", "jot"): ("allow",),
                       ("opti_oignon/rebound.py", "jot"): ("rebound",)})
        by_home = {key: _modules_named(guard.find_ungated(_census(guard, {_GATE_HOME: text, **users})))
                   for key, text in homes.items()}
    finally:
        restore()
    every = ["imported_twice", "rebound", "reexported", "relinked"]
    assert by_home == {"plain": ["rebound", "relinked"], "reassigned": every, "stub_only": every,
                       "stub_beside": ["rebound", "relinked"], "global_rebind": every, "walrus_home": every}, by_home


def test_sw57_a_program_run_by_runpy_a_temporary_object_and_a_type_checking_stub_are_read_for_what_they_are():
    program = "from opti_oignon.jot import get_jot\n\nget_jot().put('a')\n"
    runner = "import runpy\n\n\ndef go():\n    runpy.run_module('opti_oignon.tool')\n"
    holder = ("import tempfile\n\nfrom opti_oignon.jot import Jot\n\n\ndef f():\n    tmp = tempfile.TemporaryDirectory()\n"
              "    Jot(path=tmp).put('a')\n")
    stub = _JOT_TEXT.replace("def get_jot():\n    return Jot()\n",
                             "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    def get_jot():\n        return Jot()\n")
    guard, restore = _load()
    try:
        _tables(guard, exempt={("opti_oignon/tool/__main__.py", "jot"): ("script", "r"),
                               ("opti_oignon/holder.py", "jot"): ("instance", "r")})
        modules = {"opti_oignon/tool/__init__.py": "", "opti_oignon/tool/__main__.py": program,
                   "opti_oignon/holder.py": holder}
        alone = guard.find_failed_exemptions(_census(guard, modules))
        run = guard.find_failed_exemptions(_census(guard, {**modules, "opti_oignon/runner.py": runner}))
        _tables(guard)
        drift = guard.find_table_drift(guard.take_census(guard.Estate(
            Path("/nonexistent-write-census"), {"opti_oignon/__init__.py": "", _JOT: stub}, [])))
    finally:
        restore()
    temporary = (f"opti_oignon/holder.py:{_line(holder, '.put(')}: a jot write on an object the module did not build "
                 f"itself at a temporary place")
    assert alone == [temporary], alone
    assert sorted(run) == sorted([temporary, f"opti_oignon/tool/__main__.py:{_line(program, '.put(')}: a jot write "
                                             f"outside the module's __main__ block"]), run
    assert drift == [f"store jot: {_JOT} defines no function get_jot"], drift
    # A program run by keyword, by its directory, or imported as a submodule -- by its dotted name, or through a
    # fromlist that names it -- is a program no more.
    runners = {
        "keyword": "import runpy\n\n\ndef go():\n    runpy.run_module(mod_name='opti_oignon.tool')\n",
        "directory": "import runpy\n\n\ndef go():\n    runpy.run_path('opti_oignon/tool')\n",
        "dotted": "def go():\n    __import__('opti_oignon.tool.__main__')\n",
        "fromlist": "def go():\n    __import__('opti_oignon.tool', fromlist=['__main__'])\n",
    }
    guard, restore = _load()
    try:
        _tables(guard, exempt={("opti_oignon/tool/__main__.py", "jot"): ("script", "r"),
                               ("opti_oignon/holder.py", "jot"): ("instance", "r")})
        ran = {key: sorted(guard.find_failed_exemptions(_census(guard, {**modules, "opti_oignon/runner.py": text})))
               for key, text in runners.items()}
    finally:
        restore()
    outside = f"opti_oignon/tool/__main__.py:{_line(program, '.put(')}: a jot write outside the module's __main__ block"
    assert ran == {key: sorted([temporary, outside]) for key in runners}, ran
    # A program named by keyword is that program, and no other: one that runs another leaves this one a program;
    # a directory named with its trailing slash is the package.
    precise = {
        "keyword_other": "import runpy\n\n\ndef go():\n    runpy.run_module(mod_name='opti_oignon.other')\n",
        "slashed": "import runpy\n\n\ndef go():\n    runpy.run_path('opti_oignon/tool/')\n",
    }
    others = {"opti_oignon/other/__init__.py": "", "opti_oignon/other/__main__.py": "print('other')\n"}
    guard, restore = _load()
    try:
        _tables(guard, exempt={("opti_oignon/tool/__main__.py", "jot"): ("script", "r"),
                               ("opti_oignon/holder.py", "jot"): ("instance", "r")})
        ran_precise = {key: sorted(guard.find_failed_exemptions(_census(guard, {
            **modules, **others, "opti_oignon/runner.py": text}))) for key, text in precise.items()}
    finally:
        restore()
    assert ran_precise == {"keyword_other": [temporary], "slashed": sorted([temporary, outside])}, ran_precise
    # A package's fallback to None offers no other class: an object built through it is still the store's own;
    # a fallback that defines another class is one.
    fallbacks = {
        "opti_oignon/none_pkg/__init__.py": "try:\n    from opti_oignon.jot import Jot\nexcept ImportError:\n    Jot = None\n",
        "opti_oignon/alt_pkg/__init__.py": ("try:\n    from opti_oignon.jot import Jot\nexcept ImportError:\n"
                                            "    class Jot:\n        pass\n"),
    }
    builders = {f"opti_oignon/{key}_built.py": ("import tempfile\n\nfrom opti_oignon." + key + "_pkg import Jot\n\n\n"
                                                "def f():\n    Jot(path=tempfile.mkdtemp()).put('a')\n")
                for key in ("none", "alt")}
    guard, restore = _load()
    try:
        _tables(guard, exempt={(rel, "jot"): ("instance", "r") for rel in builders})
        fell_back = guard.find_failed_exemptions(_census(guard, {**fallbacks, **builders}))
    finally:
        restore()
    assert fell_back == [f"opti_oignon/alt_built.py:{_line(builders['opti_oignon/alt_built.py'], '.put(')}: a jot write "
                         f"on an object the module did not build itself at a temporary place"], fell_back
