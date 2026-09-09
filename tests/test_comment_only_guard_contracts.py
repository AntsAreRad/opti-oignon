#!/usr/bin/env python3
"""Contracts for the comment-only CI guard.

The guard proves that when internal nomenclature leaves a file, nothing that
executes leaves with it. These contracts pin the pure helpers so the
behaviour is verified independently of git:

  * C1 -- a file that sheds nomenclature from its comments alone is
    accepted: comments never reach the parser, so the shape cannot move.
  * C2 -- a file that sheds nomenclature while a runtime string edit
    leaves nomenclature behind, or touches a string that carried none, is
    refused. The one string movement the prover accepts is a purge in full.
  * C3 -- a file that sheds nomenclature AND renames an identifier is
    refused.
  * C4 -- a file whose nomenclature count is unchanged is never examined,
    whatever else moved in it. Ordinary work must not be blocked by a guard
    aimed at one dull mechanical edit.
  * C5 -- a file that GAINS nomenclature is not this guard's business; the
    public-clean guard owns that direction.
  * C6 -- the docstring of a web route handler is NOT free: the framework
    publishes it verbatim as the endpoint description of the generated API
    schema. A purge in full is accepted there on the same terms as any
    string; rewording one that carried no nomenclature is refused. An
    internal docstring in the very same file stays free.
  * C7 -- outside Python the proof is comment stripping, quote-aware: a
    comment-only edit is accepted, and an edit inside a string literal is
    refused even though it looks like a comment marker.
  * C8 -- a file whose shape cannot be established is REFUSED, never waved
    through. A prover that stays silent on what it failed to understand
    proves nothing.
  * C11 -- the docstring of a model class the schema carries is NOT free
    either. The framework publishes it as the schema description, exactly as
    it publishes a handler docstring as the endpoint description.
  * C12 -- a model class the schema does NOT carry keeps its free docstring.
    Holding every class published would refuse honest work on the internal
    models, so the published set is read, never assumed.
  * C13 -- when the published set is unknown, every class docstring is held
    published. A prover that cannot establish what ships does not get to
    assume that nothing does.
  * C14 -- a runtime string that sheds its nomenclature in full is
    accepted, and nothing else may ride along with the purge: the masked
    structure must hold, position for position.
  * C15 -- a websocket handler publishes its docstring the same way, so the
    same full purge is accepted there on the same proof.
  * C16 -- a shape difference explained in full by substituting identifiers
    that carry nomenclature for identifiers that do not is accepted.
  * C17 -- a map that is not injective is refused: two names collapsing onto
    one is not a substitution, and the collapse could hide anything.
  * C18 -- a substitution whose source carried no nomenclature is refused.
    The blade exists for one mechanical edit, not for renaming at large.
  * C19 -- a substitution whose target still carries nomenclature is refused.
  * C20 -- a structural change beside a rename is refused, and the refusal
    lists what it could not attribute. That list is the specification of the
    extraction the file still owes.
  * C21 -- the map applies inside string literals exactly as the string purge
    applies to them, and shape equality is still required afterwards.
  * C22 -- a literal the map does not explain is refused, on the same ground
    as any other unproven movement.
  * C23 -- the shape-equality proof is Python only. A non-Python file whose
    shape moves stays refused however much it looks like a rename.
  * C24 -- the published set is resolved THROUGH the map. Otherwise a renamed
    model class flips publication status mid-proof: the guard would refuse a
    docstring the diff never touched, and free one it did.
  * C25 -- an alias sequence re-sorted by a rename is refused, and named as a
    re-sort. A blade that tolerated reordering would stop guaranteeing that
    nothing else moved, which is the whole of what this guard is for.

Every input carrying nomenclature is assembled from fragments at runtime, so
the literal form never appears in this file's source and neither guard trips
on a scan of the test itself.

Local-only. Runs under pytest or via the __main__ runner. The guard script
lives under .github/, outside the importable package, and is loaded through
the shared isolation window.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate  # noqa: E402

_GUARD = "_comment_only_guard_under_contract"
_GUARD_PATH = REPO / ".github" / "scripts" / "comment_only_guard.py"

# Fragments assembled at runtime; no clear nomenclature in the source text.
_S = "S"
_CODE = _S + "347"
_OTHER = _S + "348"


def _load():
    """Load the guard through the shared window; returns (module, restore)."""
    loaded, restore = isolate(targets={_GUARD: _GUARD_PATH})
    return loaded[_GUARD], restore


# A small Python file carrying nomenclature in a comment, in an internal
# docstring, in a runtime string and in a published route docstring.
def _python_before():
    return (
        '"""Module note ' + _CODE + '."""\n'
        'import os\n'
        '\n'
        'router = os\n'
        'LABEL = "cache ' + _CODE + ' hit"\n'
        '\n'
        '\n'
        '@router.get("/x")\n'
        'def endpoint(a):\n'
        '    """Return the thing (' + _CODE + ')."""\n'
        '    # helper for ' + _CODE + ' routing\n'
        '    return os.path.join(a, "x")\n'
    )


def _typescript_before():
    return (
        '// helper for ' + _CODE + ' routing\n'
        'const url = "https://example.invalid/' + _CODE + '";\n'
        'export function g() { return url; }\n'
    )


# ---------------------------------------------------------------------------
# C1 -- shedding nomenclature from comments alone is accepted
# ---------------------------------------------------------------------------
def test_c1_comment_only_removal_is_accepted():
    guard, restore = _load()
    try:
        before = _python_before()
        after = before.replace("# helper for " + _CODE + " routing",
                               "# helper for routing")
        assert guard.debt_count(after) < guard.debt_count(before), (
            "the fixture must actually shed nomenclature"
        )
        assert guard.verdict("m.py", before, after) is None, (
            "a comment-only removal must be accepted"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# C2 -- shedding nomenclature while editing a runtime string is refused
# ---------------------------------------------------------------------------
def test_c2_runtime_string_edit_is_refused():
    guard, restore = _load()
    try:
        before = _python_before()
        purge = before.replace("# helper for " + _CODE + " routing",
                               "# helper for routing")
        # A string edit that leaves nomenclature behind rides on nothing.
        residue = purge.replace('"cache ' + _CODE + ' hit"',
                                '"cache ' + _OTHER + ' hit"')
        assert guard.debt_count(residue) < guard.debt_count(before)
        reason = guard.verdict("m.py", before, residue)
        assert reason is not None, (
            "a string still carrying nomenclature must not ride along"
        )
        assert "shape" in reason
        # A string that carried none is no purge target at all.
        untouched = purge.replace('os.path.join(a, "x")',
                                  'os.path.join(a, "y")')
        assert guard.debt_count(untouched) < guard.debt_count(before)
        assert guard.verdict("m.py", before, untouched) is not None, (
            "rewording a clean string must not ride along with a purge"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# C3 -- shedding nomenclature while renaming an identifier is refused
# ---------------------------------------------------------------------------
def test_c3_identifier_rename_is_refused():
    guard, restore = _load()
    try:
        before = _python_before()
        after = before.replace("# helper for " + _CODE + " routing",
                               "# helper for routing")
        after = after.replace("LABEL", "TAG")
        assert guard.verdict("m.py", before, after) is not None, (
            "a rename must not ride along with a comment purge"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# C4 -- an unchanged count means the file is never examined
# ---------------------------------------------------------------------------
def test_c4_unchanged_count_is_not_examined():
    guard, restore = _load()
    try:
        before = _python_before()
        # A real code change that keeps every nomenclature line intact.
        after = before.replace('return os.path.join(a, "x")',
                               'return os.path.join(a, "y", "z")')
        assert guard.debt_count(after) == guard.debt_count(before)
        assert guard.verdict("m.py", before, after) is None, (
            "ordinary work must never be blocked by this guard"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# C5 -- a file that gains nomenclature belongs to the other guard
# ---------------------------------------------------------------------------
def test_c5_added_nomenclature_is_not_this_guard():
    guard, restore = _load()
    try:
        before = _python_before()
        after = before.replace("import os",
                               "import os  # added in " + _OTHER)
        after = after.replace("LABEL", "TAG")
        assert guard.debt_count(after) > guard.debt_count(before)
        assert guard.verdict("m.py", before, after) is None, (
            "the rising direction is the public-clean guard's business"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# C6 -- a published route description is shape, an internal docstring is not
# ---------------------------------------------------------------------------
def test_c6_published_route_description_is_not_free():
    guard, restore = _load()
    try:
        before = _python_before()
        # Internal module docstring: free.
        internal = before.replace('"""Module note ' + _CODE + '."""',
                                  '"""Module note."""')
        assert guard.debt_count(internal) < guard.debt_count(before)
        assert guard.verdict("m.py", before, internal) is None, (
            "an internal docstring is not a shipped surface"
        )
        # Route handler docstring: published, yet a purge in full is a purge.
        published = before.replace('"""Return the thing (' + _CODE + ')."""',
                                   '"""Return the thing."""')
        assert guard.debt_count(published) < guard.debt_count(before)
        assert guard.verdict("m.py", before, published) is None, (
            "a full purge of an endpoint description is the dull edit itself"
        )
        # Rewording a published description that carried none is refused.
        clean = before.replace('"""Return the thing (' + _CODE + ')."""',
                               '"""Return the thing."""')
        reworded = clean.replace("# helper for " + _CODE + " routing",
                                 "# helper for routing")
        reworded = reworded.replace('"""Return the thing."""',
                                    '"""Return the whole thing."""')
        assert guard.debt_count(reworded) < guard.debt_count(clean)
        assert guard.verdict("m.py", clean, reworded) is not None, (
            "an endpoint description that carried none must not move"
        )
    finally:
        restore()


def _python_ws_before():
    return (
        'import os\n'
        '\n'
        'router = os\n'
        '\n'
        '\n'
        '@router.websocket("/w")\n'
        'def stream(a):\n'
        '    """Feed the thing (' + _CODE + ')."""\n'
        '    return a\n'
    )


# ---------------------------------------------------------------------------
# C14 -- a full purge of a runtime string is accepted, alone
# ---------------------------------------------------------------------------
def test_c14_full_string_purge_is_accepted_alone():
    guard, restore = _load()
    try:
        before = _python_before()
        purge = before.replace("# helper for " + _CODE + " routing",
                               "# helper for routing")
        purge = purge.replace('"cache ' + _CODE + ' hit"', '"cache hit"')
        assert guard.debt_count(purge) < guard.debt_count(before)
        assert guard.verdict("m.py", before, purge) is None, (
            "a string that sheds its nomenclature in full is the dull edit"
        )
        # The same purge with a rename riding along is refused.
        riding = purge.replace("LABEL", "TAG")
        assert guard.verdict("m.py", before, riding) is not None, (
            "the masked structure must hold, position for position"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# C15 -- a websocket description purges on the same proof
# ---------------------------------------------------------------------------
def test_c15_websocket_docstring_purge_is_accepted():
    guard, restore = _load()
    try:
        before = _python_ws_before()
        after = before.replace('"""Feed the thing (' + _CODE + ')."""',
                               '"""Feed the thing."""')
        assert guard.debt_count(after) < guard.debt_count(before)
        assert guard.verdict("m.py", before, after) is None, (
            "a full purge of a websocket description is the dull edit"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# C7 -- outside Python, the proof is quote-aware comment stripping
# ---------------------------------------------------------------------------
def test_c7_non_python_proof_is_quote_aware():
    guard, restore = _load()
    try:
        before = _typescript_before()
        comment_only = before.replace("// helper for " + _CODE + " routing",
                                      "// helper for routing")
        assert guard.debt_count(comment_only) < guard.debt_count(before)
        assert guard.verdict("a.ts", before, comment_only) is None, (
            "a comment-only edit outside Python must be accepted"
        )
        # The same token inside a string literal is NOT a comment.
        in_string = before.replace("// helper for " + _CODE + " routing",
                                   "// helper for routing")
        in_string = in_string.replace("example.invalid/" + _CODE,
                                      "example.invalid/x")
        assert guard.verdict("a.ts", before, in_string) is not None, (
            "an edit inside a string literal must not pass as a comment"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# C8 -- a shape that cannot be established is refused, never assumed
# ---------------------------------------------------------------------------
def test_c8_unestablished_shape_is_refused():
    guard, restore = _load()
    try:
        before = "# note " + _CODE + "\nvalue = (1,\n"  # unbalanced on purpose
        after = "# note\nvalue = (1,\n"
        assert guard.debt_count(after) < guard.debt_count(before)
        reason = guard.verdict("broken.py", before, after)
        assert reason is not None, (
            "a file the prover cannot parse must be refused, not waved on"
        )
        assert "could not be established" in reason
    finally:
        restore()


# ---------------------------------------------------------------------------
# C9 -- a hash glued to the token before it does not open a comment
# ---------------------------------------------------------------------------
def test_c9_glued_hash_does_not_open_a_comment():
    guard, restore = _load()
    try:
        before = (
            "while [[ $# -gt 0 ]]; do  # argument loop " + _CODE + "\n"
            "  shift\n"
            "done\n"
        )
        # Nomenclature leaves the trailing comment AND the loop bound moves.
        after = (
            "while [[ $# -gt 1 ]]; do  # argument loop\n"
            "  shift\n"
            "done\n"
        )
        assert guard.debt_count(after) < guard.debt_count(before)
        assert guard.verdict("run.sh", before, after) is not None, (
            "a hash with no whitespace before it is not a comment opener; "
            "the code after it must stay visible to the prover"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# C10 -- the two honest hash positions still open a comment
# ---------------------------------------------------------------------------
def test_c10_line_start_and_spaced_hash_still_comment():
    guard, restore = _load()
    try:
        at_start = "# header " + _CODE + "\nvalue: 1\n"
        at_start_after = "# header\nvalue: 1\n"
        assert guard.debt_count(at_start_after) < guard.debt_count(at_start)
        assert guard.verdict("a.yaml", at_start, at_start_after) is None, (
            "a hash at the start of a line opens a comment"
        )

        spaced = "value: 1  # note " + _CODE + "\n"
        spaced_after = "value: 1  # note\n"
        assert guard.debt_count(spaced_after) < guard.debt_count(spaced)
        assert guard.verdict("a.yaml", spaced, spaced_after) is None, (
            "a hash preceded by whitespace opens a comment"
        )

        # And the value itself is still shape, not comment.
        moved = "value: 2  # note\n"
        assert guard.verdict("a.yaml", spaced, moved) is not None, (
            "an edit to the value must not pass as a comment edit"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _model_file(note):
    """A model class and an ordinary class, each carrying a docstring."""
    return (
        'from pydantic import BaseModel\n'
        '\n'
        '\n'
        'class Widget(BaseModel):\n'
        f'    """{note}"""\n'
        '    size: int\n'
        '\n'
        '\n'
        'class Helper:\n'
        '    """An internal helper."""\n'
        '    pass\n'
    )


def test_c11_published_model_description_is_not_free():
    guard, restore = _load()
    try:
        before = _model_file("A widget as the client sees it.")
        after = _model_file("A widget as the client sees it today.")
        published = {"Widget"}
        assert (guard.python_shape(before, published_models=published)
                != guard.python_shape(after, published_models=published)), \
            "a published schema description moved without moving the shape"
        # The very same edit on a class the schema does not carry is free.
        assert (guard.python_shape(before, published_models=set())
                == guard.python_shape(after, published_models=set()))
    finally:
        restore()


def test_c12_unpublished_model_keeps_a_free_docstring():
    guard, restore = _load()
    try:
        # No published set given: the guard reads the recorded digest, which
        # cannot name a class invented here.
        before = _model_file("An internal note.")
        after = _model_file("An internal note, reworded.")
        assert guard.python_shape(before) == guard.python_shape(after), \
            "an unpublished model docstring was held published"
    finally:
        restore()


def test_c13_unknown_published_set_holds_every_class():
    guard, restore = _load()
    try:
        before = _model_file("A widget as the client sees it.")
        after = _model_file("A widget as the client sees it today.")
        assert (guard.python_shape(before, published_models=None)
                != guard.python_shape(after, published_models=None)), \
            "an unknown published set was read as an empty one"
    finally:
        restore()


# ---------------------------------------------------------------------------
# C16-C25 -- the proven-rename blade
#
# A file that sheds nomenclature may also move its executable shape, but only
# when the whole difference is explained by substituting identifiers that
# carry nomenclature for identifiers that do not. Every fragment carrying
# nomenclature is assembled at runtime, as everywhere else in this file.
# ---------------------------------------------------------------------------
def _carry():
    """An identifier carrying nomenclature."""
    return _CODE + "Cache"


def _carry_low():
    """The same nomenclature in the form that appears inside literals."""
    return (_CODE + "_cache").lower()


def _rename_before():
    """Nomenclature in a comment and in an identifier used twice."""
    return (
        'import os\n'
        '\n'
        '# note for ' + _CODE + ' routing\n'
        'class ' + _carry() + 'Stats:\n'
        '    """An internal note."""\n'
        '    size = 3\n'
        '\n'
        '\n'
        'def build(entry):\n'
        '    return os.path.join(entry, str(' + _carry() + 'Stats.size))\n'
    )


def _purged(text):
    """The same file with the nomenclature-bearing comment removed."""
    return text.replace('# note for ' + _CODE + ' routing\n', '')


def _literal_before():
    """Nomenclature in a comment, in an identifier, and inside a string."""
    return (
        'import os\n'
        '\n'
        '# note for ' + _CODE + ' routing\n'
        + _carry_low() + ' = "' + _carry_low() + '_hit"\n'
        '\n'
        '\n'
        'def build(entry):\n'
        '    return os.path.join(entry, ' + _carry_low() + ')\n'
    )


def _published_model(name, note):
    """A model class the schema may carry, under a chosen name."""
    return (
        'from pydantic import BaseModel\n'
        '\n'
        '\n'
        'class ' + name + '(BaseModel):\n'
        '    """' + note + '"""\n'
        '    size: int\n'
    )


def test_c16_a_proven_rename_is_accepted():
    guard, restore = _load()
    try:
        before = _rename_before()
        after = _purged(before).replace(_carry() + "Stats", "SemCacheStats")
        assert guard.debt_count(after) < guard.debt_count(before), (
            "the fixture must actually shed nomenclature"
        )
        assert guard.python_shape(before) != guard.python_shape(after), (
            "the fixture must actually move the shape, or the acceptance "
            "proves nothing that shape equality did not already prove"
        )
        assert guard.verdict("m.py", before, after) is None, (
            "a substitution explaining the whole shape difference must be "
            "accepted"
        )
    finally:
        restore()


def test_c17_a_non_injective_map_is_refused():
    guard, restore = _load()
    try:
        before = (
            '# note for ' + _CODE + ' routing\n'
            'class ' + _carry() + 'Stats:\n'
            '    size = 3\n'
            '\n'
            '\n'
            'class ' + _OTHER + 'CacheStats:\n'
            '    size = 4\n'
        )
        after = (_purged(before)
                 .replace(_carry() + "Stats", "SemCacheStats")
                 .replace(_OTHER + "CacheStats", "SemCacheStats"))
        assert guard.debt_count(after) < guard.debt_count(before)
        reason = guard.verdict("m.py", before, after)
        assert reason is not None, (
            "two names collapsing onto one is not a substitution"
        )
        assert "injective" in reason, reason
    finally:
        restore()


def test_c18_renaming_an_identifier_that_carried_nothing_is_refused():
    guard, restore = _load()
    try:
        before = _rename_before()
        after = (_purged(before)
                 .replace(_carry() + "Stats", "SemCacheStats")
                 .replace("build", "assemble"))
        assert guard.debt_count(after) < guard.debt_count(before)
        reason = guard.verdict("m.py", before, after)
        assert reason is not None, (
            "a clean identifier must not be renamed under cover of a purge"
        )
        assert "carried no nomenclature" in reason, reason
    finally:
        restore()


def test_c19_renaming_onto_a_name_that_still_carries_is_refused():
    guard, restore = _load()
    try:
        before = _rename_before()
        after = _purged(before).replace(_carry() + "Stats",
                                        _OTHER + "CacheStats")
        assert guard.debt_count(after) < guard.debt_count(before), (
            "the fixture must still shed nomenclature overall"
        )
        reason = guard.verdict("m.py", before, after)
        assert reason is not None, (
            "renaming nomenclature onto more nomenclature is not a purge"
        )
        assert "still carries nomenclature" in reason, reason
    finally:
        restore()


def test_c20_structural_change_beside_a_rename_is_refused_and_listed():
    guard, restore = _load()
    try:
        before = _rename_before()
        after = (_purged(before).replace(_carry() + "Stats", "SemCacheStats")
                 + '\n\ndef extra():\n    return 1\n')
        assert guard.debt_count(after) < guard.debt_count(before)
        reason = guard.verdict("m.py", before, after)
        assert reason is not None, (
            "a new function must not ride along with a rename"
        )
        assert "not attributable" in reason, reason
        assert "FunctionDef" in reason, (
            "the refusal must name the residue, not merely report one: "
            + str(reason)
        )
    finally:
        restore()


def test_c21_the_map_applies_inside_string_literals():
    guard, restore = _load()
    try:
        before = _literal_before()
        after = _purged(before).replace(_carry_low(), "semcache")
        assert guard.debt_count(after) < guard.debt_count(before)
        assert guard.python_shape(before) != guard.python_shape(after), (
            "the fixture must actually move the shape"
        )
        assert guard.verdict("m.py", before, after) is None, (
            "a literal whose change the map explains is part of the rename"
        )
    finally:
        restore()


def test_c22_a_literal_the_map_does_not_explain_is_refused():
    guard, restore = _load()
    try:
        before = _literal_before()
        after = (_purged(before)
                 .replace('"' + _carry_low() + '_hit"', '"semcache_miss"')
                 .replace(_carry_low(), "semcache"))
        assert guard.debt_count(after) < guard.debt_count(before)
        reason = guard.verdict("m.py", before, after)
        assert reason is not None, (
            "a literal edit beyond the substitution must not ride along"
        )
        assert "not explained" in reason, reason
    finally:
        restore()


def test_c23_outside_python_a_moved_shape_stays_refused():
    guard, restore = _load()
    try:
        before = (
            '// note for ' + _CODE + ' routing\n'
            'const ' + _carry() + 'Stats = 3;\n'
            'export function g() { return ' + _carry() + 'Stats; }\n'
        )
        after = (before.replace('// note for ' + _CODE + ' routing\n', '')
                 .replace(_carry() + "Stats", "SemCacheStats"))
        assert guard.debt_count(after) < guard.debt_count(before)
        assert guard.verdict("a.ts", before, after) is not None, (
            "the shape-equality proof is Python only; a non-Python file "
            "whose shape moves stays refused whatever it looks like"
        )
    finally:
        restore()


def test_c24_the_published_set_is_resolved_through_the_map():
    guard, restore = _load()
    try:
        note = "A widget as the client sees it."
        before = _published_model(_carry() + "Stats", note)
        after = _published_model("SemCacheStats", note)
        published = {"SemCacheStats"}
        assert guard.rename_equivalent(
            before, after, published_models=published) is None, (
            "a renamed published model whose description never moved must "
            "not be refused for a docstring the diff never touched"
        )
        moved = _published_model("SemCacheStats", note + " Today.")
        assert guard.rename_equivalent(
            before, moved, published_models=published) is not None, (
            "a published description that did move must still be refused"
        )
    finally:
        restore()


def test_c25_a_rename_induced_alias_resort_is_refused_as_a_resort():
    guard, restore = _load()
    try:
        before = (
            '# note for ' + _CODE + ' routing\n'
            'from os import (path, ' + _carry() + 'Stats)\n'
        )
        after = 'from os import (AaaCache, path)\n'
        assert guard.debt_count(after) < guard.debt_count(before)
        reason = guard.verdict("m.py", before, after)
        assert reason is not None, (
            "a re-sorted alias sequence is not an identifier substitution"
        )
        assert "re-sorted" in reason, (
            "the re-sort must be named as such, not reported as any "
            "difference: " + str(reason)
        )
    finally:
        restore()


def _run_all():
    tests = [
        ("C1 comment-only removal accepted",
         test_c1_comment_only_removal_is_accepted),
        ("C2 runtime string edit refused",
         test_c2_runtime_string_edit_is_refused),
        ("C3 identifier rename refused", test_c3_identifier_rename_is_refused),
        ("C4 unchanged count not examined",
         test_c4_unchanged_count_is_not_examined),
        ("C5 added nomenclature not this guard",
         test_c5_added_nomenclature_is_not_this_guard),
        ("C6 published route description is shape",
         test_c6_published_route_description_is_not_free),
        ("C7 non-Python proof is quote-aware",
         test_c7_non_python_proof_is_quote_aware),
        ("C8 unestablished shape refused",
         test_c8_unestablished_shape_is_refused),
        ("C9 glued hash is not a comment opener",
         test_c9_glued_hash_does_not_open_a_comment),
        ("C10 honest hash positions still comment",
         test_c10_line_start_and_spaced_hash_still_comment),
        ("C11 published model description is shape",
         test_c11_published_model_description_is_not_free),
        ("C12 unpublished model docstring free",
         test_c12_unpublished_model_keeps_a_free_docstring),
        ("C13 unknown published set holds every class",
         test_c13_unknown_published_set_holds_every_class),
        ("C14 full string purge accepted alone",
         test_c14_full_string_purge_is_accepted_alone),
        ("C15 websocket description purges the same",
         test_c15_websocket_docstring_purge_is_accepted),
        ("C16 proven rename accepted", test_c16_a_proven_rename_is_accepted),
        ("C17 non-injective map refused",
         test_c17_a_non_injective_map_is_refused),
        ("C18 clean source rename refused",
         test_c18_renaming_an_identifier_that_carried_nothing_is_refused),
        ("C19 carrying target refused",
         test_c19_renaming_onto_a_name_that_still_carries_is_refused),
        ("C20 structural residue listed",
         test_c20_structural_change_beside_a_rename_is_refused_and_listed),
        ("C21 map applies inside literals",
         test_c21_the_map_applies_inside_string_literals),
        ("C22 unexplained literal refused",
         test_c22_a_literal_the_map_does_not_explain_is_refused),
        ("C23 non-Python moved shape refused",
         test_c23_outside_python_a_moved_shape_stays_refused),
        ("C24 published set resolved through the map",
         test_c24_the_published_set_is_resolved_through_the_map),
        ("C25 alias re-sort named as a re-sort",
         test_c25_a_rename_induced_alias_resort_is_refused_as_a_resort),
    ]
    passed = 0
    for label, fn in tests:
        try:
            fn()
            print(f"PASS  {label}")
            passed += 1
        except Exception:  # noqa: BLE001 -- report and continue
            print(f"FAIL  {label}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
