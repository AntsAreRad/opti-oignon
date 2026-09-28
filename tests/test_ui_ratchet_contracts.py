#!/usr/bin/env python3
"""Contracts for the frontend ratchets and the ledger engine that holds them.

A ratchet counts one pattern per file of the frontend and keeps the counts in
a ledger, a literal ``{path: count}`` dict in this file. The engine lives once,
in ``tests/_frontend.py`` (``check_ledger``), and every ratchet goes through
it. Its rules are proven here on synthetic repositories built under
``$TMPDIR``, never on the real tree, so each rule is seen failing on input
made to break it:

  * RX1 -- a ledger entry higher than its value at ``HEAD`` fails. The value
    at ``HEAD`` is read from the committed test file with ``git show`` and
    ``ast``, never from the working tree, where the raised value sits.
  * RX2 -- a file that is not in the ledger counts 0: an unlisted file with a
    non-zero count fails, and an unlisted clean file does not.
  * RX3 -- an entry above its file's count is stale and fails with "lower the
    ledger", and so does an entry whose file is gone.
  * RX4 -- the file list is never empty, and it is the listing rule
    ``git ls-files --cached --others --exclude-standard``: a file being
    written and not yet staged is counted, an ignored file is not.
  * RX5 -- a renamed file carries its entry only when ``git diff -M HEAD``
    shows the rename and the entry did not rise. The rename reader sees the
    listing's untracked files (marked intent-to-add in a scratch index), so
    a move made without ``git mv`` is carried as git's similarity pairs it;
    a new file carries nothing, and neither does a file moved and then
    written again at its old path.
  * RX6 -- a listed file above its entry fails.
  * RX7 -- a count function that reads zero on its positive fixture fails,
    even over a clean tree: a probe that has gone blind is red, not a quiet
    zero.
  * RX8 -- a ledger that is not at ``HEAD`` under its name in its test file
    answers to the ledger it was: renamed, or moved to another test file,
    it is compared with the one ``HEAD`` holds that shares its files and is
    gone; a new ledger beside a kept one is born; several gone is refused.

The ratchets that follow hold the real tree, over the listing rule of
``tests/_frontend.py``. Each ledger is born equal to the count its own
function measures, so each is green at birth and proven by its blades: one
instance added to a file the ledger holds reads red, and so does each
exclusion dropped, form made blind or threshold moved. Every census carries a
standing positive fixture; a census with several forms asserts that each one
is counted, and a census with an exclusion asserts that a negative sample
reads 0.

  * UR1 -- hand-made buttons outside the primitives (``lib/ds/``): a
    ``<button`` tag, or an element given ``role="button"``.
  * UR2 -- hand-made ``<input``, ``<select`` and ``<textarea`` outside the
    primitives.
  * UR3 -- ``style=`` attributes and ``style:`` directives in components.
    Geometry a script writes at run time (a width an action sets, a
    position a placement library sets) is outside this census: it is a
    measured value, not a style written in the markup.
  * UR4 -- type under 12 px, in px, rem or em: ``text-[N<unit>]`` classes,
    ``font-size`` (declaration or directive), the ``font`` shorthand, and a
    text-scale token declared under 12 px (once, at its declaration).
  * UR5 -- ``rgb(`` and ``rgba(`` literals in components.
  * UR6 -- hover-only reveals: ``group-hover:<utility>`` that no focus
    variant sets too, in the same ``class=`` value (or quoted string).
  * UR7 -- raw ``fetch(`` outside the API layer (``lib/api/``), bare or
    through the global object.
  * UR8 -- French comment lines: the comment-only guard's comment spans,
    judged line by line by the language guard's ``is_french``, both guards
    loaded through the shared isolation window; every kind of file listed
    is read, and a plain reading blind to quotes is the floor under it.
  * UR9 -- ``setInterval(`` sites.
  * UR10 -- symbol glyphs (U+2600-27BF, U+2B00-2BFF), raw, as an HTML
    numeric or named reference, a script or CSS escape, or a code point
    built from a number literal.
  * UR11 -- ``var(`` reads of a border token (``--oo-bd-*``) or of an alias
    that carries one, the aliases derived from the declarations.
  * UR12 -- capitals and wide tracking: ``uppercase``, small capitals, the
    wide tracking classes and token, and a positive ``letter-spacing``.
  * UR13 -- lines between rows: the ``border-t``, ``border-b``,
    ``border-y``, ``divide-x`` and ``divide-y`` classes, and the CSS that
    draws the same line (``border-top``, ``border-bottom``,
    ``border-block*``) as a declaration, an inline style or a directive.
  * UR14 -- gradients, except the ``linear-gradient(currentColor 0 0 ...)``
    drawing idiom, and the Tailwind gradient utilities.
  * UR15 -- ``location.reload(``.
  * UR16 -- raw ``fetch(`` in the API layer outside its client
    (``lib/api/client.ts``).
  * UR17 -- every file the listing holds under the source is of a kind at
    least one census reads, or is named as not read with its reason: a
    file of a new kind is never a zero for every ratchet at once.

Local-only (the public distribution ships no tests). Runs under pytest or
the __main__ runner. Needs git; the synthetic repositories are built with the
maintainer's git configuration set aside, and nothing is ever signed.
"""

import html.entities
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _frontend import REPO, check_ledger, files, pattern_count, read  # noqa: E402
from _isolation import isolate  # noqa: E402

# Seconds each contract may take on this machine, read back by the ladder from the junit file.
BUDGET_S = {
    "test_rx1_an_entry_above_its_value_at_head_fails": 1.0,
    "test_rx2_an_unlisted_file_with_a_nonzero_count_fails": 1.0,
    "test_rx3_a_stale_entry_fails_with_lower_the_ledger": 1.0,
    "test_rx4_the_listing_is_never_empty_and_counts_untracked_unignored_files": 1.0,
    "test_rx5_a_rename_carries_its_entry_only_when_git_shows_it_and_it_did_not_rise": 1.0,
    "test_rx6_a_listed_file_above_its_entry_fails": 1.0,
    "test_rx7_a_count_function_blind_to_its_fixture_fails": 1.0,
    "test_rx8_a_ledger_renamed_or_moved_since_head_answers_to_the_one_it_was": 1.0,
    "test_ur1_hand_made_buttons_outside_the_primitives_never_rise": 1.0,
    "test_ur2_hand_made_fields_outside_the_primitives_never_rise": 1.0,
    "test_ur3_style_attributes_never_rise": 1.0,
    "test_ur4_type_under_twelve_pixels_never_rises": 1.0,
    "test_ur5_rgb_literals_in_components_never_rise": 1.0,
    "test_ur6_hover_only_reveals_never_rise": 1.0,
    "test_ur7_raw_fetch_outside_the_api_layer_never_rises": 1.0,
    "test_ur8_french_comment_lines_never_rise": 1.0,
    "test_ur9_interval_timers_never_rise": 1.0,
    "test_ur10_symbol_glyphs_never_rise": 1.0,
    "test_ur11_border_token_references_never_rise": 1.0,
    "test_ur12_capitals_and_wide_tracking_never_rise": 1.0,
    "test_ur13_row_borders_and_dividers_never_rise": 1.0,
    "test_ur14_gradients_never_rise": 1.0,
    "test_ur15_page_reloads_never_rise": 1.0,
    "test_ur16_raw_fetch_in_the_api_layer_outside_its_client_never_rises": 1.0,
    "test_ur17_every_file_under_the_source_is_of_a_kind_the_censuses_read": 1.0,
}

_LEDGER_NAME = "LEDGER"
_TEST_FILE = "tests/test_census_contracts.py"
_A = "frontend/src/lib/a.svelte"
_B = "frontend/src/lib/b.svelte"
_C = "frontend/src/lib/c.svelte"
_MARK = "MARK"
_COUNT = pattern_count(r"\bMARK\b")

# Variables that point git at another repository or index; a synthetic
# repository must answer for itself.
_GIT_LOCATORS = (
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_NAMESPACE",
)


def _git(root, *args):
    env = {k: v for k, v in os.environ.items() if k not in _GIT_LOCATORS}
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    subprocess.run(
        [
            "git", "-C", str(root),
            "-c", "user.name=ratchet", "-c", "user.email=ratchet@example.invalid",
            "-c", "commit.gpgsign=false", *args,
        ],
        check=True, capture_output=True, env=env,
    )


def _body(marks, filler=12):
    """A source file with ``marks`` counted tokens and enough stable lines
    for git's rename detection to see the same file after a small edit."""
    lines = [f'<p class="line-{i}">stable text {i}</p>' for i in range(filler)]
    lines.extend(f"<span>{_MARK}</span>" for _ in range(marks))
    return "\n".join(lines) + "\n"


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _ledger_source(ledger):
    return f"{_LEDGER_NAME} = {ledger!r}\n"


def _repo(tmp, sources, ledger, ignore=""):
    """A committed repository: the sources, and the ledger in its test file."""
    root = Path(tmp) / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    for rel, text in sources.items():
        _write(root, rel, text)
    _write(root, _TEST_FILE, _ledger_source(ledger))
    if ignore:
        _write(root, ".gitignore", ignore)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "birth")
    return root


def _check(root, ledger, count_fn=_COUNT, fixture=_MARK):
    """The engine over the synthetic tree, as a ratchet contract calls it.

    The working-tree test file carries the same ledger the call passes, as it
    does for a real ratchet; the engine must read HEAD's, not this one.
    """
    _write(root, _TEST_FILE, _ledger_source(ledger))
    return check_ledger(
        _LEDGER_NAME, ledger, count_fn, fixture,
        test_file=root / _TEST_FILE, root=root, suffixes=(".svelte",),
    )


# ---------------------------------------------------------------------------
# RX1 -- an entry never rises above its value at HEAD
# ---------------------------------------------------------------------------
def test_rx1_an_entry_above_its_value_at_head_fails():
    with tempfile.TemporaryDirectory() as tmp:
        root = _repo(tmp, {_A: _body(2)}, {_A: 2})
        census = _check(root, {_A: 2})
        assert census.head == "compared", census.head
        assert census.counts == {_A: 2}, census.counts
        _write(root, _A, _body(3))
        with pytest.raises(AssertionError) as caught:
            _check(root, {_A: 3})
        message = str(caught.value)
        assert _A in message and "above its value at HEAD" in message, message


# ---------------------------------------------------------------------------
# RX2 -- an unlisted file counts 0
# ---------------------------------------------------------------------------
def test_rx2_an_unlisted_file_with_a_nonzero_count_fails():
    with tempfile.TemporaryDirectory() as tmp:
        root = _repo(tmp, {_A: _body(1)}, {_A: 1})
        _write(root, _C, _body(0))
        census = _check(root, {_A: 1})
        assert census.counts.get(_C, 0) == 0, census.counts
        _write(root, _B, _body(1))
        with pytest.raises(AssertionError) as caught:
            _check(root, {_A: 1})
        message = str(caught.value)
        assert _B in message and "not in the ledger" in message, message


# ---------------------------------------------------------------------------
# RX3 -- an entry above the count is stale: lower the ledger
# ---------------------------------------------------------------------------
def test_rx3_a_stale_entry_fails_with_lower_the_ledger():
    with tempfile.TemporaryDirectory() as tmp:
        root = _repo(tmp, {_A: _body(2), _B: _body(1)}, {_A: 2, _B: 1})
        _write(root, _A, _body(1))
        with pytest.raises(AssertionError) as caught:
            _check(root, {_A: 2, _B: 1})
        message = str(caught.value)
        assert _A in message and "lower the ledger" in message, message
        # A file that is gone counts 0, so its entry is stale too.
        _write(root, _A, _body(2))
        (root / _B).unlink()
        with pytest.raises(AssertionError) as caught:
            _check(root, {_A: 2, _B: 1})
        message = str(caught.value)
        assert _B in message and "lower the ledger" in message, message


# ---------------------------------------------------------------------------
# RX4 -- the listing: never empty, untracked counted, ignored not
# ---------------------------------------------------------------------------
def test_rx4_the_listing_is_never_empty_and_counts_untracked_unignored_files():
    ignored = "frontend/src/lib/ignored.svelte"
    with tempfile.TemporaryDirectory() as tmp:
        # The ledger at HEAD owes one instance in a file not yet written.
        root = _repo(
            tmp, {"frontend/src/lib/only.ts": "export const x = 1;\n"}, {_A: 1},
            ignore=ignored + "\n",
        )
        with pytest.raises(AssertionError) as caught:
            _check(root, {_A: 1})
        assert "empty" in str(caught.value), str(caught.value)
        # Written, not staged, not ignored: listed and counted.
        _write(root, _A, _body(1))
        census = _check(root, {_A: 1})
        assert _A in census.files, census.files
        assert census.counts == {_A: 1}, census.counts
        # Ignored: never listed, whatever it holds.
        _write(root, ignored, _body(5))
        census = _check(root, {_A: 1})
        assert ignored not in census.files, census.files


# ---------------------------------------------------------------------------
# RX5 -- a rename carries its entry only when git shows it, and never higher
# ---------------------------------------------------------------------------
def test_rx5_a_rename_carries_its_entry_only_when_git_shows_it_and_it_did_not_rise():
    with tempfile.TemporaryDirectory() as tmp:
        root = _repo(tmp, {_A: _body(2)}, {_A: 2})
        _git(root, "mv", _A, _B)
        census = _check(root, {_B: 2})
        assert census.counts == {_B: 2}, census.counts
        _write(root, _B, _body(3))
        with pytest.raises(AssertionError) as caught:
            _check(root, {_B: 3})
        message = str(caught.value)
        assert _B in message and "rename" in message, message
    with tempfile.TemporaryDirectory() as tmp:
        root = _repo(tmp, {_A: _body(2)}, {_A: 2})
        # A new file, like nothing deleted, is not a rename.
        other = "\n".join(f"<b>other {i}</b>" for i in range(12))
        _write(root, _C, f"{other}\n<span>{_MARK}</span>\n")
        with pytest.raises(AssertionError) as caught:
            _check(root, {_A: 2, _C: 1})
        message = str(caught.value)
        assert _C in message and "not a rename" in message, message
    with tempfile.TemporaryDirectory() as tmp:
        root = _repo(tmp, {_A: _body(2)}, {_A: 2})
        # Moved without git mv, the index left as it was: the listing counts
        # the untracked file, and the rename reader sees it too.
        (root / _A).rename(root / _C)
        census = _check(root, {_C: 2})
        assert census.counts == {_C: 2}, census.counts
        _write(root, _C, _body(3))
        with pytest.raises(AssertionError) as caught:
            _check(root, {_C: 3})
        message = str(caught.value)
        assert _C in message and "rename" in message, message
    with tempfile.TemporaryDirectory() as tmp:
        root = _repo(tmp, {_A: _body(2)}, {_A: 2})
        # Moved with git mv, then written again at the old path: the old
        # path is not deleted, so the new one is no rename and carries
        # nothing, and HEAD's entry is never claimed twice.
        _git(root, "mv", _A, _B)
        _write(root, _A, _body(2))
        with pytest.raises(AssertionError) as caught:
            _check(root, {_A: 2, _B: 2})
        message = str(caught.value)
        assert _B in message and "not a rename" in message, message


# ---------------------------------------------------------------------------
# RX6 -- no listed file above its entry
# ---------------------------------------------------------------------------
def test_rx6_a_listed_file_above_its_entry_fails():
    with tempfile.TemporaryDirectory() as tmp:
        root = _repo(tmp, {_A: _body(1)}, {_A: 1})
        _write(root, _A, _body(2))
        with pytest.raises(AssertionError) as caught:
            _check(root, {_A: 1})
        message = str(caught.value)
        assert _A in message and "above its entry" in message, message


# ---------------------------------------------------------------------------
# RX7 -- a blind count function is red, even over a clean tree
# ---------------------------------------------------------------------------
def test_rx7_a_count_function_blind_to_its_fixture_fails():
    with tempfile.TemporaryDirectory() as tmp:
        root = _repo(tmp, {_A: _body(0)}, {})
        census = _check(root, {})
        assert census.counts == {}, census.counts
        blind = pattern_count(r"\bNOT_THE_MARK\b")
        with pytest.raises(AssertionError) as caught:
            _check(root, {}, count_fn=blind)
        assert "blind" in str(caught.value), str(caught.value)


# ---------------------------------------------------------------------------
# RX8 -- a ledger renamed or moved since HEAD answers to the one it was
# ---------------------------------------------------------------------------
def _hold_as(root, test_file, name, ledger, count_fn=_COUNT, fixture=_MARK):
    """The engine over the synthetic tree, with the ledger assigned to
    ``name`` in ``test_file`` (written first, as a real ratchet has it)."""
    _write(root, test_file, f"{name} = {ledger!r}\n")
    return check_ledger(
        name, ledger, count_fn, fixture,
        test_file=root / test_file, root=root, suffixes=(".svelte",),
    )


def test_rx8_a_ledger_renamed_or_moved_since_head_answers_to_the_one_it_was():
    with tempfile.TemporaryDirectory() as tmp:
        root = _repo(tmp, {_A: _body(2)}, {_A: 2})
        _write(root, _A, _body(3))
        # Renamed: the name HEAD knew is gone, and the new one carries a rise.
        with pytest.raises(AssertionError) as caught:
            _hold_as(root, _TEST_FILE, "RENAMED", {_A: 3})
        message = str(caught.value)
        assert _A in message and "above its value at HEAD" in message, message
        assert f"{_TEST_FILE}:{_LEDGER_NAME}" in message, message
    with tempfile.TemporaryDirectory() as tmp:
        root = _repo(tmp, {_A: _body(2)}, {_A: 2})
        _write(root, _A, _body(3))
        # Moved: the test file renamed on disk, its ledger with it.
        (root / _TEST_FILE).unlink()
        with pytest.raises(AssertionError) as caught:
            _hold_as(root, "tests/test_census_moved_contracts.py", _LEDGER_NAME, {_A: 3})
        message = str(caught.value)
        assert _A in message and "above its value at HEAD" in message, message
        assert f"{_TEST_FILE}:{_LEDGER_NAME}" in message, message
    with tempfile.TemporaryDirectory() as tmp:
        root = _repo(tmp, {_A: _body(2) + "<i>OTHER</i>\n" * 5}, {_A: 2})
        # Born beside a ledger that is kept: a new census over the same
        # files answers to nothing but today's count.
        (root / _TEST_FILE).write_text(
            _ledger_source({_A: 2}) + "NEWBORN = {" + repr(_A) + ": 5}\n",
            encoding="utf-8",
        )
        census = check_ledger(
            "NEWBORN", {_A: 5}, pattern_count(r"\bOTHER\b"), "OTHER",
            test_file=root / _TEST_FILE, root=root, suffixes=(".svelte",),
        )
        assert census.head == "born", census.head
        assert census.counts == {_A: 5}, census.counts
    with tempfile.TemporaryDirectory() as tmp:
        root = _repo(tmp, {_A: _body(2)}, {_A: 2})
        (root / _TEST_FILE).write_text(
            "FIRST = {" + repr(_A) + ": 2}\nSECOND = {" + repr(_A) + ": 2}\n",
            encoding="utf-8",
        )
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", "two ledgers")
        # Both gone and one new name over their files: which one it was
        # cannot be told, so it is refused.
        with pytest.raises(AssertionError) as caught:
            _hold_as(root, _TEST_FILE, "MERGED", {_A: 2})
        message = str(caught.value)
        assert "several ledgers" in message and "FIRST" in message, message


# ===========================================================================
# The ratchets on the real tree. Each ledger is born equal to today's count,
# so a ratchet is green at birth; its blade adds one instance to a file the
# ledger already holds and reads red.
# ===========================================================================
# The kinds of file each census reads. Every kind the source holds is read by
# at least one census (UR17), so a file of a new kind is never a quiet zero.
_MARKUP = (".svelte",)
_CODE = (".svelte", ".ts", ".js", ".mjs", ".cjs")
_STYLED = _CODE + (".css", ".scss")
_EVERY = _STYLED + (".html",)
_PRIMITIVES = "frontend/src/lib/ds/"
_API = "frontend/src/lib/api"
_API_CLIENT = "frontend/src/lib/api/client.ts"


def _hold(name, count_fn, fixture, **listing):
    """Hold one census of the real tree to the ledger assigned to ``name`` here."""
    return check_ledger(
        name, globals()[name], count_fn, fixture, test_file=__file__, **listing,
    )


# ---------------------------------------------------------------------------
# UR1 -- hand-made buttons outside the primitives
# ---------------------------------------------------------------------------
# A <button> tag, or any element given the button role (a hand-made button
# without the native one's keyboard behaviour). A CSS attribute selector
# naming the role is not an element.
_BUTTONS = pattern_count(
    r"<button\b|(?<![\w:\[-])role\s*=\s*(?:\{\s*)?[\"'`]?button(?![\w-])"
)


def test_ur1_hand_made_buttons_outside_the_primitives_never_rise():
    _hold(
        "UR1_LEDGER", _BUTTONS, '<button type="button">Save</button>',
        suffixes=_MARKUP, exclude=(_PRIMITIVES,),
    )
    forms = (
        '<div role="button" tabindex="0"></div><span role={"button"}></span>\n'
        "<li role='button'></li><b role=button></b>\n"
    )
    assert _BUTTONS("frontend/src/fixture.svelte", forms) == 4, (
        "an element given the button role is a hand-made button, in each spelling"
    )
    not_buttons = (
        '<style>[role="button"] { cursor: pointer; }</style>\n'
        '<div role="buttonbar"></div><div data-role="button"></div>\n'
    )
    assert _BUTTONS("frontend/src/fixture.svelte", not_buttons) == 0, (
        "a selector naming the role, another role or a data attribute is not a button"
    )


# ---------------------------------------------------------------------------
# UR2 -- hand-made fields outside the primitives: input, select, textarea
# ---------------------------------------------------------------------------
_FIELDS = pattern_count(r"<(?:input|select|textarea)\b")


def test_ur2_hand_made_fields_outside_the_primitives_never_rise():
    census = _hold(
        "UR2_LEDGER", _FIELDS,
        '<input type="text" /><select></select><textarea></textarea>',
        suffixes=_MARKUP, exclude=(_PRIMITIVES,),
    )
    assert census.fixture == 3, f"each of the three field tags is counted: {census.fixture}"


# ---------------------------------------------------------------------------
# UR3 -- style attributes in components
# ---------------------------------------------------------------------------
# A style attribute, or a Svelte style directive (style:color=, style:--x=,
# or the shorthand style:color), which sets one property inline the same way.
_STYLE_ATTRIBUTES = pattern_count(r"\sstyle(?:\s*=|:[A-Za-z-])")


def test_ur3_style_attributes_never_rise():
    _hold(
        "UR3_LEDGER", _STYLE_ATTRIBUTES, '<div style="gap: 4px"></div>',
        suffixes=_MARKUP,
    )
    directives = (
        '<p style:color="red" style:--gap={gap} style:opacity></p>\n'
        '<p style ="gap: 1px"></p>\n'
    )
    assert _STYLE_ATTRIBUTES("frontend/src/fixture.svelte", directives) == 4, (
        "each style directive and a spaced attribute are counted"
    )
    assert _STYLE_ATTRIBUTES(
        "frontend/src/fixture.svelte", "<style>a { font-style: italic; }</style>",
    ) == 0, "a CSS property ending in -style is not an attribute"


# ---------------------------------------------------------------------------
# UR4 -- type under 12 px, in px, rem or em, in each spelling
# ---------------------------------------------------------------------------
_UNITS = r"(\d*\.?\d+)(px|rem|em)(?![\w-])"
_TEXT_SIZE = re.compile(r"(?<![\w-])text-\[(\d*\.?\d+)(px|rem|em)\]")
_FONT_SIZE = re.compile(
    r"(?:font-size\s*:|(?<![\w-])style:font-size\s*=\s*[{\"'`]*)\s*" + _UNITS
)
_FONT_SHORTHAND = re.compile(r"(?<![\w-])font\s*:\s*([^;{}\"'<>\n]*)")
_SIZE_TOKEN = re.compile(
    r"--oo-text-[\w-]+\s*:\s*(?:calc\(\s*)?" + _UNITS
)
# rem and em are read over a 16 px parent.
_SMALLEST = {"px": 12.0, "rem": 0.75, "em": 0.75}


def _small(number, unit):
    return float(number) < _SMALLEST[unit]


def _small_type(path, text):
    """Font sizes under 12 px, in px, rem or em: ``text-[N<unit>]``
    classes, ``font-size`` declarations and ``style:font-size`` directives,
    the size in a ``font`` shorthand, and a text-scale token declared under
    12 px, counted once, at its declaration (its uses follow it; a use of a
    token is not counted). A size at 12 px (0.75rem, 0.75em) or above is not
    counted."""
    found = sum(1 for match in _TEXT_SIZE.finditer(text) if _small(*match.groups()))
    found += sum(1 for match in _FONT_SIZE.finditer(text) if _small(*match.groups()))
    found += sum(1 for match in _SIZE_TOKEN.finditer(text) if _small(*match.groups()))
    for match in _FONT_SHORTHAND.finditer(text):
        size = re.search(_UNITS, match.group(1))
        if size and _small(*size.groups()):
            found += 1
    return found


def test_ur4_type_under_twelve_pixels_never_rises():
    census = _hold(
        "UR4_LEDGER", _small_type,
        '<p class="text-[10px]"></p>\n'
        "<style>a { font-size: 0.7rem; } b { font-size: 11px; }</style>\n",
        suffixes=_STYLED,
    )
    assert census.fixture == 3, f"each of the three forms is counted: {census.fixture}"
    at_the_line = (
        '<p class="text-[12px] text-xs"></p>\n'
        "<style>a { font-size: 0.75rem; } b { font-size: 12px; } "
        "c { font-size: 0.8rem; } d { font-size: var(--oo-text-xs); }</style>\n"
    )
    assert _small_type("frontend/src/fixture.svelte", at_the_line) == 0, (
        "a size at 12 px or above, or carried by a token, is not counted"
    )
    other_forms = (
        '<p class="text-[0.7rem] text-[0.6em]" style:font-size="10px"></p>\n'
        "<style>a { font-size: 0.7em; } b { font: 600 10px/1.2 sans-serif; }\n"
        ":root { --oo-text-2xs: calc(11px * var(--oo-type-scale, 1)); }</style>\n"
    )
    assert _small_type("frontend/src/fixture.svelte", other_forms) == 6, (
        "rem and em classes, the directive, em, the shorthand and a small "
        "token's declaration are each counted"
    )
    level_forms = (
        "<style>a { font: inherit; } b { font-size: 0.75em; } "
        "c { font: 12px/1 sans-serif; } "
        ":root { --oo-text-xs: calc(12px * var(--oo-type-scale, 1)); "
        "--oo-text-primary: var(--oo-fg-primary); }</style>\n"
    )
    assert _small_type("frontend/src/fixture.svelte", level_forms) == 0, (
        "a shorthand without a small size, 0.75em, and a token at 12 px or "
        "without a size are not counted"
    )


# ---------------------------------------------------------------------------
# UR5 -- rgb and rgba literals in components
# ---------------------------------------------------------------------------
_RGB = pattern_count(r"rgba?\(\s*\d")


def test_ur5_rgb_literals_in_components_never_rise():
    _hold(
        "UR5_LEDGER", _RGB, '<div style="color: rgba(0, 0, 0, 0.4)"></div>',
        suffixes=_MARKUP,
    )


# ---------------------------------------------------------------------------
# UR6 -- hover-only reveals: a group-hover utility no focus variant sets
# ---------------------------------------------------------------------------
_HOVER = re.compile(r"(?<![\w-])group-hover:([^\s\"'`{}<>]+)")
_FOCUS_VARIANTS = (
    "group-focus-within", "group-focus-visible", "group-focus",
    "focus-within", "focus-visible", "focus",
)
_CLASS_OPEN = re.compile(r"(?<![\w:-])class\s*=\s*")


def _balanced(text, start):
    """The end of the braced expression opening at ``start`` (quotes and
    nested braces skipped), or the end of the text."""
    depth, index, quote = 0, start, None
    while index < len(text):
        char = text[index]
        if quote:
            if char == "\\":
                index += 1
            elif char == quote:
                quote = None
        elif char in "\"'`":
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return len(text)


def _class_values(text):
    """The spans of every ``class=`` value: a quoted string, or a braced
    expression to its matching brace."""
    spans = []
    for match in _CLASS_OPEN.finditer(text):
        start = match.end()
        if start >= len(text):
            continue
        opener = text[start]
        if opener in "\"'":
            end = text.find(opener, start + 1)
            spans.append((start, len(text) if end < 0 else end + 1))
        elif opener == "{":
            spans.append((start, _balanced(text, start)))
    return spans


def _string_around(text, position):
    """The quoted string on the line around ``position``, or that line."""
    line_start = text.rfind("\n", 0, position) + 1
    line_end = text.find("\n", position)
    line_end = len(text) if line_end < 0 else line_end
    for quote in "\"'`":
        opening = text.rfind(quote, line_start, position)
        if opening >= 0:
            closing = text.find(quote, position, line_end)
            if closing >= 0:
                return opening, closing + 1
    return line_start, line_end


def _hover_only(path, text):
    """``group-hover:<utility>`` variants whose utility no focus variant sets
    too (``group-focus-within:``, ``group-focus-visible:``, ``group-focus:``,
    ``focus-within:``, ``focus-visible:`` or ``focus:`` with the same
    utility), so what they reveal is out of reach of the keyboard. The
    focus variant must sit in the same ``class=`` value, or, outside one, in
    the same quoted string on the line; a focus variant setting another
    utility (``focus:outline-none``) is no alternative."""
    spans = _class_values(text)
    found = 0
    for match in _HOVER.finditer(text):
        utility = match.group(1)
        scope = next(
            (span for span in spans if span[0] <= match.start() < span[1]), None,
        ) or _string_around(text, match.start())
        within = text[scope[0]:scope[1]]
        if not any(
            re.search(rf"(?<![\w-]){variant}:{re.escape(utility)}(?![\w/.\[-])", within)
            for variant in _FOCUS_VARIANTS
        ):
            found += 1
    return found


def test_ur6_hover_only_reveals_never_rise():
    census = _hold(
        "UR6_LEDGER", _hover_only,
        '<div class="opacity-0 group-hover:opacity-100"></div>',
        suffixes=_CODE,
    )
    assert census.fixture == 1, census.fixture
    reachable = (
        '<div class="opacity-0 group-hover:opacity-100\n'
        '\tgroup-focus-within:opacity-100"></div>'
    )
    assert _hover_only("frontend/src/fixture.svelte", reachable) == 0, (
        "a reveal the focus reaches too is not hover-only"
    )
    unreachable = (
        '<button class="opacity-0 group-hover:opacity-100 focus:outline-none">\n'
        '<span class="invisible group-hover:visible focus-visible:ring-2"></span>\n'
        "<div class=\"p-2 {open ? 'a' : ''} opacity-0 group-hover:opacity-100\"></div>\n"
        "<input class=\"focus:ring-2\" placeholder=\"it's\" />\n"
        '<div class="opacity-0 group-hover:opacity-100"></div>'
        '<i class="group-focus-within:opacity-100"></i>\n'
    )
    assert _hover_only("frontend/src/fixture.svelte", unreachable) == 4, (
        "a focus variant that sets another utility, or sits on another "
        "element, is no keyboard alternative"
    )
    in_script = (
        "const reveal = 'opacity-0 group-hover:opacity-100 "
        "group-focus-within:opacity-100';\n"
        "const bare = 'hidden group-hover:flex';\n"
    )
    assert _hover_only("frontend/src/fixture.ts", in_script) == 1, (
        "in a script, the quoted string holds the class list"
    )


# ---------------------------------------------------------------------------
# UR7 -- raw fetch outside the API layer
# ---------------------------------------------------------------------------
# The global fetch, bare or through the global object (window., globalThis.,
# self.); a method of another object (api.fetch, this.fetch) is not it.
_RAW_FETCH = pattern_count(r"(?<![\w.$])(?:(?:window|globalThis|self)\??\.)?fetch\(")


def test_ur7_raw_fetch_outside_the_api_layer_never_rises():
    _hold(
        "UR7_LEDGER", _RAW_FETCH, "const reply = await fetch('/api/health');",
        suffixes=_CODE, exclude=(_API + "/",),
    )
    globals_ = "window.fetch(a); globalThis.fetch(b); self.fetch(c); window?.fetch(d);"
    assert _RAW_FETCH("frontend/src/fixture.ts", globals_) == 4, (
        "the global fetch through the global object is counted, once per call"
    )
    methods = "api.fetch(a); this.fetch(b); prefetch(c); x.window.fetch(d); $fetch(e);"
    assert _RAW_FETCH("frontend/src/fixture.ts", methods) == 0, (
        "a method or a function of another name is not the global fetch"
    )


# ---------------------------------------------------------------------------
# UR8 -- French comment lines, read with the repository's own guards
# ---------------------------------------------------------------------------
_COMMENT_GUARD = "_comment_only_guard_under_ratchet"
_LANGUAGE_GUARD = "_public_language_guard_under_ratchet"
_GUARDS = REPO / ".github" / "scripts"


def _french_comment_lines(comments, language):
    """Comment lines that read as French: the comment spans the comment-only
    guard computes, judged line by line by the language guard's ``is_french``.
    A line's comment text is every span on it; code and strings are not."""

    def count(path, text):
        suffix = PurePosixPath(path).suffix
        if suffix in comments._MARKUP_LIKE:
            spans = comments._comment_spans(text, markup=True)
        elif suffix in comments._C_LIKE:
            spans = comments._comment_spans(text)
        else:
            return 0
        # The span rows count newlines only, as the guard's scanner does.
        lines = text.split("\n")
        found = 0
        for row, cuts in spans.items():
            if row > len(lines):
                continue
            line = lines[row - 1]
            if language.is_french(" ".join(line[low:high] for low, high in cuts)):
                found += 1
        return found

    return count


# A plain reading of comments, blind to strings: block and markup comments
# anywhere, and a line comment from a "//" that does not end a URL scheme.
_PLAIN_BLOCK = re.compile(r"/\*.*?\*/|<!--.*?-->", re.DOTALL)
_PLAIN_LINE = re.compile(r"(?<![:\\])//(.*)$", re.MULTILINE)


def _plain_french_comment_lines(language):
    """French comment lines by a plain reading that knows no quotes: the
    floor under the guard's quote-aware scanner. A quote the scanner takes
    for an open string (an apostrophe in markup text, a quote inside a
    regular expression literal) hides the comments after it from the
    scanner and not from this reading."""

    def count(path, text):
        rows = {}
        for match in [*_PLAIN_BLOCK.finditer(text), *_PLAIN_LINE.finditer(text)]:
            row = text.count("\n", 0, match.start()) + 1
            for offset, piece in enumerate(match.group(0).split("\n")):
                rows.setdefault(row + offset, []).append(piece)
        return sum(1 for pieces in rows.values() if language.is_french(" ".join(pieces)))

    return count


def test_ur8_french_comment_lines_never_rise():
    loaded, restore = isolate(targets={
        _COMMENT_GUARD: _GUARDS / "comment_only_guard.py",
        _LANGUAGE_GUARD: _GUARDS / "public_language_guard.py",
    })
    try:
        count = _french_comment_lines(loaded[_COMMENT_GUARD], loaded[_LANGUAGE_GUARD])
        prose = "Retourne la liste des fichiers pour le cache"
        census = _hold(
            "UR8_LEDGER", count,
            f"<script>\n\t// {prose}\n\t/* {prose} */\n</script>\n<!-- {prose} -->\n",
            suffixes=_EVERY,
        )
        assert census.fixture == 3, (
            f"line, block and markup comments are each read: {census.fixture}"
        )
        # The French string shares its line with a short English comment:
        # only the comment's own text is judged.
        not_comments = (
            f'const label = "{prose}"; // the label\n'
            "// Return the list of files for the cache\n"
        )
        assert count("frontend/src/fixture.ts", not_comments) == 0, (
            "French in a string, or an English comment, is not a French comment line"
        )
        # Every kind of file the census lists is read: one comment each.
        blind = [
            suffix for suffix in _EVERY
            if count(
                f"frontend/src/fixture{suffix}",
                f"<!-- {prose} -->\n" if suffix in (".svelte", ".html")
                else f"/* {prose} */\n",
            ) < 1
        ]
        assert not blind, f"the census reads no comment in these kinds of file: {blind}"
        # Floor: no file holds more French comment lines by a plain reading
        # than the census counts in it.
        plain = _plain_french_comment_lines(loaded[_LANGUAGE_GUARD])
        hidden = {}
        for path in census.files:
            seen = plain(path, read(path))
            if seen > census.counts.get(path, 0):
                hidden[path] = (seen, census.counts.get(path, 0))
        assert not hidden, (
            f"a plain reading finds more French comment lines than the census "
            f"counts (plain, census): the scanner hides comments: {hidden}"
        )
        hiding = (
            "<p>L'outil</p> <!-- Retourne la liste des fichiers pour le cache -->\n"
        )
        assert plain("frontend/src/fixture.svelte", hiding) == 1, (
            "the floor reads a comment after an apostrophe in markup text"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# UR9 -- interval timers
# ---------------------------------------------------------------------------
_INTERVALS = pattern_count(r"setInterval\(")


def test_ur9_interval_timers_never_rise():
    _hold(
        "UR9_LEDGER", _INTERVALS, "const timer = setInterval(poll, 15000);",
        suffixes=_CODE,
    )


# ---------------------------------------------------------------------------
# UR10 -- symbol glyphs, in each spelling that draws them
# ---------------------------------------------------------------------------
_GLYPH_RANGES = ((0x2600, 0x27BF), (0x2B00, 0x2BFF))
_RAW_GLYPH = re.compile(
    "[" + "".join(f"{chr(low)}-{chr(high)}" for low, high in _GLYPH_RANGES) + "]"
)
_REFERENCE = re.compile(r"&#(?:[xX]([0-9a-fA-F]+)|([0-9]+));")
# A script escape: a backslash, the letter u, then four hex digits or a
# braced code point. The backslash is built, never typed.
_ESCAPE = re.compile(re.escape(chr(92)) + r"u(?:\{([0-9a-fA-F]+)\}|([0-9a-fA-F]{4}))")
# A CSS escape: a backslash and one to six hex digits (content: "...").
_CSS_ESCAPE = re.compile(re.escape(chr(92)) + r"([0-9a-fA-F]{1,6})")
# A named character reference, resolved through the HTML5 table.
_NAMED = re.compile(r"&([A-Za-z][A-Za-z0-9]*;)")
# A code point built at run time from a number literal.
_FROM_CODE = re.compile(r"fromC(?:odePoint|harCode)\(\s*(0[xX][0-9a-fA-F]+|\d+)\s*[,)]")


def _in_glyph_ranges(point):
    return any(low <= point <= high for low, high in _GLYPH_RANGES)


def _symbol_glyphs(path, text):
    """Code points in the two ranges of ``_GLYPH_RANGES`` (the Miscellaneous
    Symbols and Dingbats blocks, and the Miscellaneous Symbols and Arrows
    block: check marks, crosses, gear, stars, trigram), in each spelling
    that draws the same glyph: raw, an HTML numeric or named reference, a
    script string escape, a CSS escape, or ``String.fromCodePoint`` /
    ``fromCharCode`` over a number literal."""
    found = len(_RAW_GLYPH.findall(text))
    for match in _REFERENCE.finditer(text):
        hexadecimal, decimal = match.groups()
        point = int(hexadecimal, 16) if hexadecimal else int(decimal)
        if _in_glyph_ranges(point):
            found += 1
    for match in _ESCAPE.finditer(text):
        braced, plain = match.groups()
        if _in_glyph_ranges(int(braced or plain, 16)):
            found += 1
    for match in _CSS_ESCAPE.finditer(text):
        if _in_glyph_ranges(int(match.group(1), 16)):
            found += 1
    for match in _NAMED.finditer(text):
        found += sum(
            1 for char in html.entities.html5.get(match.group(1), "")
            if _in_glyph_ranges(ord(char))
        )
    for match in _FROM_CODE.finditer(text):
        number = match.group(1)
        point = int(number, 16) if number[:2] in ("0x", "0X") else int(number)
        if _in_glyph_ranges(point):
            found += 1
    return found


def test_ur10_symbol_glyphs_never_rise():
    sample = (
        f"<span>{chr(0x2605)}</span><span>&#10003;</span>\n"
        f"<script>const mark = '{chr(92)}u2699';</script>\n"
    )
    census = _hold("UR10_LEDGER", _symbol_glyphs, sample, suffixes=_EVERY)
    assert census.fixture == 3, f"each of the three forms is counted: {census.fixture}"
    outside = (
        f"<span>{chr(0x2192)}</span><span>&#8212;</span><span>&#x25B6;</span>\n"
        f"<script>const dash = '{chr(92)}u2014';</script>\n"
    )
    assert _symbol_glyphs("frontend/src/fixture.svelte", outside) == 0, (
        "a code point outside the two ranges is not counted"
    )
    spelled = (
        f"<style>a::after {{ content: '{chr(92)}2713'; }} "
        f"b::before {{ content: '{chr(92)}002605 '; }}</style>\n"
        "<span>&check;</span><span>&starf;</span>\n"
        "<script>const g = String.fromCodePoint(0x2699) + "
        "String.fromCharCode(10003);</script>\n"
    )
    assert _symbol_glyphs("frontend/src/fixture.svelte", spelled) == 6, (
        "CSS escapes, named references and code points built from a number "
        "are each counted"
    )
    spelled_outside = (
        f"<style>a::after {{ content: '{chr(92)}2014'; }}</style>\n"
        "<span>&rarr;</span><span>&middot;</span><span>&amp;</span>\n"
        "<script>const b = String.fromCharCode(byte) + "
        "String.fromCharCode(65);</script>\n"
    )
    assert _symbol_glyphs("frontend/src/fixture.svelte", spelled_outside) == 0, (
        "a named reference, escape or code point outside the ranges is not counted"
    )


# ---------------------------------------------------------------------------
# UR11 -- border token reads, directly or through an alias
# ---------------------------------------------------------------------------
_BORDER_PREFIX = "--oo-bd-"
_DECLARATION = re.compile(r"(?<![\w-])(--[\w-]+)\s*:\s*([^;{}]*)")
_VAR_NAME = re.compile(r"var\(\s*(--[\w-]+)")


def _border_aliases(texts):
    """Custom properties that carry a border token: declared with a value
    that reads one (``--oo-border: var(--oo-bd-default)``), or reads such an
    alias, to the fixed point."""
    declarations = [
        (match.group(1), set(_VAR_NAME.findall(match.group(2))))
        for text in texts for match in _DECLARATION.finditer(text)
    ]
    aliases = set()
    while True:
        grown = {
            name for name, reads in declarations
            if name not in aliases and not name.startswith(_BORDER_PREFIX)
            and any(ref.startswith(_BORDER_PREFIX) or ref in aliases for ref in reads)
        }
        if not grown:
            return aliases
        aliases |= grown


def _border_token_uses(aliases):
    """``var(`` reads of a border token (``--oo-bd-*``) or of an alias that
    carries one; a declaration is not a read, and its value's reads count."""

    def count(path, text):
        return sum(
            1 for ref in _VAR_NAME.findall(text)
            if ref.startswith(_BORDER_PREFIX) or ref in aliases
        )

    return count


def test_ur11_border_token_references_never_rise():
    aliases = _border_aliases(read(path) for path in files(_STYLED))
    census = _hold(
        "UR11_LEDGER", _border_token_uses(aliases),
        "<style>a { border: 1px solid var(--oo-bd-subtle); }</style>",
        suffixes=_STYLED,
    )
    assert census.fixture == 1, census.fixture
    sample = (
        ":root { --oo-line: var(--oo-bd-subtle); --oo-rule: 1px solid var(--oo-line); "
        "--oo-ink: var(--oo-fg-primary); }\n"
        "a { border: 1px solid var( --oo-bd-default); border-top: var(--oo-rule); "
        "color: var(--oo-ink); outline: 1px solid var(--oo-line); }\n"
    )
    found = _border_aliases([sample])
    assert found == {"--oo-line", "--oo-rule"}, (
        f"an alias of a border token, and an alias of that alias, are found: {found}"
    )
    assert _border_token_uses(found)("frontend/src/fixture.css", sample) == 5, (
        "a spaced read and every read of an alias are counted; another token is not"
    )


# ---------------------------------------------------------------------------
# UR12 -- capitals and wide tracking
# ---------------------------------------------------------------------------
_UPPERCASE = re.compile(r"(?<![\w-])(?:uppercase|(?:all-)?small-caps)(?![\w-])")
_TRACKING_CLASS = re.compile(
    r"(?<![\w-])tracking-(?:wide|wider|widest|\[(\d*\.?\d+)[a-z%]*\]"
    r"|\[var\(\s*--oo-tracking-wide[\w-]*\s*\)\])(?![\w-])"
)
_LETTER_SPACING = re.compile(
    r"(?:letter-spacing\s*:|(?<![\w-])style:letter-spacing\s*=\s*[{\"'`]*)"
    r"\s*([^;\"'`}\n]*)"
)
_LEADING_NUMBER = re.compile(r"\+?(\d*\.?\d+)")
_WIDE_TOKEN = re.compile(r"var\(\s*--oo-tracking-wide")


def _caps_and_tracking(path, text):
    """Capitals and wide tracking: ``uppercase`` and ``small-caps`` (class
    or CSS value), the ``tracking-wide*`` classes, a positive arbitrary
    tracking or the wide tracking token as a class, and ``letter-spacing``
    (declaration or ``style:`` directive) set to a positive length or to the
    wide tracking token. Negative or zero spacing, the tight and normal
    tokens, and a token's own declaration are not counted."""
    found = len(_UPPERCASE.findall(text))
    for match in _TRACKING_CLASS.finditer(text):
        if match.group(1) is None or float(match.group(1)) > 0:
            found += 1
    for match in _LETTER_SPACING.finditer(text):
        value = match.group(1).strip()
        number = _LEADING_NUMBER.match(value)
        if (number and float(number.group(1)) > 0) or _WIDE_TOKEN.match(value):
            found += 1
    return found


def test_ur12_capitals_and_wide_tracking_never_rise():
    census = _hold(
        "UR12_LEDGER", _caps_and_tracking,
        '<p class="uppercase tracking-widest tracking-[0.1em]"></p>\n'
        "<style>a { letter-spacing: 0.05em; } "
        "b { letter-spacing: var(--oo-tracking-wide); }</style>\n",
        suffixes=_STYLED,
    )
    assert census.fixture == 5, f"each of the five forms is counted: {census.fixture}"
    level = (
        '<p class="normal-case tracking-tight tracking-[-0.02em]"></p>\n'
        "<style>a { letter-spacing: -0.01em; } b { letter-spacing: 0; } "
        "c { letter-spacing: var(--oo-tracking-tight); } "
        "d { text-transform: capitalize; } :root { --oo-tracking-wide: 0.025em; }"
        "</style>\n"
    )
    assert _caps_and_tracking("frontend/src/fixture.svelte", level) == 0, (
        "level or tight spacing, and a token's declaration, are not counted"
    )
    other_forms = (
        '<p class="tracking-[var(--oo-tracking-wide)]" style:letter-spacing="0.08em"></p>\n'
        "<style>a { font-variant: all-small-caps; } "
        "b { font-variant-caps: small-caps; }</style>\n"
    )
    assert _caps_and_tracking("frontend/src/fixture.svelte", other_forms) == 4, (
        "the wide token as a class, the directive and small capitals are each counted"
    )


# ---------------------------------------------------------------------------
# UR13 -- lines between rows: the classes, and the CSS that draws them
# ---------------------------------------------------------------------------
_ROW_LINES = re.compile(r"(?<![\w-])(?:border-[tby]|divide-[xy])(?![a-z])")
_ROW_SIDES = r"border-(?:top|bottom|block(?:-start|-end)?)(?:-width|-style)?"
_ROW_PROPERTIES = re.compile(
    rf"(?:(?<![\w-]){_ROW_SIDES}\s*:|(?<![\w-])style:{_ROW_SIDES}\s*=\s*[{{\"'`]*)"
    r"\s*([^;\"'`}\n]*)"
)
_NO_LINE = frozenset({"none", "0", "0px", "hidden"})


def _row_lines(path, text):
    """Lines between rows: the ``border-t``, ``border-b``, ``border-y``,
    ``divide-x`` and ``divide-y`` classes with their variants, and the CSS
    that draws the same line (``border-top``, ``border-bottom`` and the
    logical ``border-block*``, with their ``-width`` and ``-style``) as a
    declaration, an inline style or a ``style:`` directive. A side border,
    a colour alone, and a line removed (``none``, ``0``, ``hidden``) are not
    counted."""
    found = len(_ROW_LINES.findall(text))
    for match in _ROW_PROPERTIES.finditer(text):
        value = re.sub(r"\s*!\s*important\s*$", "", match.group(1).strip()).lower()
        if value and value not in _NO_LINE:
            found += 1
    return found


def test_ur13_row_borders_and_dividers_never_rise():
    census = _hold(
        "UR13_LEDGER", _row_lines,
        '<li class="border-b"></li><ul class="divide-y"></ul>',
        suffixes=_STYLED,
    )
    assert census.fixture == 2, f"borders and dividers are each counted: {census.fixture}"
    lookalikes = (
        '<p class="border-l border-x border-yellow-700 border-teal-500"></p>\n'
        "<style>a { border-left: 1px solid; border-top-color: red; "
        "box-sizing: border-box; border-bottom: none; border-top: 0; "
        "border-block-end: hidden !important; }</style>\n"
    )
    assert _row_lines("frontend/src/fixture.svelte", lookalikes) == 0, (
        "side borders, colours alone and removed lines are not counted"
    )
    drawn = (
        '<li class="border-y"></li>\n'
        "<style>a { border-bottom: 1px solid var(--oo-bd-subtle); "
        "b { border-block-end: 1px solid; } c { border-top-width: 2px; }</style>\n"
        '<p style="border-top: 1px solid red"></p>'
        '<p style:border-bottom="1px solid"></p>\n'
    )
    assert _row_lines("frontend/src/fixture.svelte", drawn) == 6, (
        "border-y, and the line drawn by CSS as a declaration, an inline style "
        "or a directive, are each counted"
    )


# ---------------------------------------------------------------------------
# UR14 -- gradients, except the currentColor drawing idiom
# ---------------------------------------------------------------------------
_GRADIENT = re.compile(
    r"(?<![\w-])(?!linear-gradient\(\s*currentColor\s+0\s+0)[\w-]*gradient\("
)
# The Tailwind gradient utilities (bg-gradient-to-*, and bg-linear-*,
# bg-radial, bg-conic); their colour stops (from-, via-, to-) are the same
# gradient and are not counted again.
_GRADIENT_CLASS = re.compile(r"(?<![\w-])bg-(?:gradient|linear|radial|conic)(?![a-z])")


def _gradients(path, text):
    """Every ``*gradient(`` except ``linear-gradient(currentColor 0 0 ...)``,
    which paints a drawn shape with the text colour and shades nothing, and
    every Tailwind gradient utility."""
    return len(_GRADIENT.findall(text)) + len(_GRADIENT_CLASS.findall(text))


def test_ur14_gradients_never_rise():
    census = _hold(
        "UR14_LEDGER", _gradients,
        "<style>a { background: radial-gradient(circle, var(--oo-acc-500) 0%, "
        "transparent 70%); }</style>",
        suffixes=_STYLED,
    )
    assert census.fixture == 1, census.fixture
    idiom = (
        "<style>a { background-image: linear-gradient(currentColor 0 0); }\n"
        "b { background-image: linear-gradient(\n"
        "\t\tcurrentColor 0 0,\n\t\tcurrentColor 0 0\n\t); }</style>\n"
    )
    assert _gradients("frontend/src/fixture.svelte", idiom) == 0, (
        "the currentColor drawing idiom is not a gradient"
    )
    utilities = (
        '<div class="bg-gradient-to-r from-amber-500 via-rose-400 to-rose-500"></div>\n'
        '<div class="bg-radial bg-conic-180 bg-linear-to-b"></div>\n'
    )
    assert _gradients("frontend/src/fixture.svelte", utilities) == 4, (
        "each Tailwind gradient utility is counted once, its stops not again"
    )
    assert _gradients(
        "frontend/src/fixture.svelte",
        '<div class="bg-gray-100 bg-linearity bg-conical-x bg-radials"></div>',
    ) == 0, "a background utility that is not a gradient is not counted"


# ---------------------------------------------------------------------------
# UR15 -- page reloads
# ---------------------------------------------------------------------------
_RELOADS = pattern_count(r"location\.reload\(")


def test_ur15_page_reloads_never_rise():
    _hold(
        "UR15_LEDGER", _RELOADS, "window.location.reload();", suffixes=_CODE,
    )


# ---------------------------------------------------------------------------
# UR16 -- raw fetch in the API layer outside its client
# ---------------------------------------------------------------------------
def test_ur16_raw_fetch_in_the_api_layer_outside_its_client_never_rises():
    _hold(
        "UR16_LEDGER", _RAW_FETCH,
        ("frontend/src/lib/api/fixture.ts", "const reply = await fetch(url);"),
        suffixes=_CODE, within=_API, exclude=(_API_CLIENT,),
    )


# ---------------------------------------------------------------------------
# UR17 -- every file under the source is of a kind the censuses read
# ---------------------------------------------------------------------------
# Kinds of file the source may hold that no census reads, each with its
# reason. Empty: every kind it holds today is read.
_NOT_READ = {}


def _unread(listed):
    return sorted(
        path for path in listed
        if not path.endswith(_EVERY) and not path.endswith(tuple(_NOT_READ))
    )


def test_ur17_every_file_under_the_source_is_of_a_kind_the_censuses_read():
    listed = files(None)
    unread = _unread(listed)
    assert not unread, (
        f"files of a kind no census reads, so every ratchet reads them as "
        f"zero: {unread}; read the kind in the censuses it concerns, or name "
        f"it in _NOT_READ with its reason"
    )
    kinds = {PurePosixPath(path).suffix for path in listed}
    assert {".svelte", ".ts"} <= kinds, f"the listing reads no source: {sorted(kinds)}"
    planted = [*listed, "frontend/src/lib/icon.svg", "frontend/src/lib/util.jsx"]
    assert _unread(planted) == ["frontend/src/lib/icon.svg", "frontend/src/lib/util.jsx"], (
        "a kind no census reads is named"
    )


# ===========================================================================
# The ledgers. Each is born equal to the count its contract measures on the
# tree; an entry only ever goes down, and a file improved lowers its entry.
# ===========================================================================
UR1_LEDGER = {
    'frontend/src/lib/components/chat/BranchExplorer.svelte': 11,
    'frontend/src/lib/components/chat/BranchTreeNodeItem.svelte': 1,
    'frontend/src/lib/components/chat/ChatControlBar.svelte': 8,
    'frontend/src/lib/components/chat/ChatInput.svelte': 5,
    'frontend/src/lib/components/chat/ChatMessage.svelte': 2,
    'frontend/src/lib/components/chat/ContextPanel.svelte': 2,
    'frontend/src/lib/components/chat/CorrectionIndicator.svelte': 1,
    'frontend/src/lib/components/chat/ExportDialog.svelte': 1,
    'frontend/src/lib/components/chat/FeedbackWidget.svelte': 2,
    'frontend/src/lib/components/chat/FileUpload.svelte': 2,
    'frontend/src/lib/components/chat/ModelSelector.svelte': 3,
    'frontend/src/lib/components/chat/PresetSelector.svelte': 3,
    'frontend/src/lib/components/chat/ProjectContextBadge.svelte': 1,
    'frontend/src/lib/components/chat/ProjectLinker.svelte': 3,
    'frontend/src/lib/components/chat/ReasoningDisplay.svelte': 1,
    'frontend/src/lib/components/chat/RoutingIndicator.svelte': 1,
    'frontend/src/lib/components/chat/ScrollToBottomFab.svelte': 1,
    'frontend/src/lib/components/chat/SearchResults.svelte': 1,
    'frontend/src/lib/components/chat/ToolCallDisplay.svelte': 2,
    'frontend/src/lib/components/health/HealthDashboard.svelte': 1,
    'frontend/src/lib/components/layout/AppShell.svelte': 1,
    'frontend/src/lib/components/panels/AnalyticsDashboard.svelte': 2,
    'frontend/src/lib/components/panels/AnswerVerifier.svelte': 1,
    'frontend/src/lib/components/panels/ArtifactPanel.svelte': 9,
    'frontend/src/lib/components/panels/CacheStatsPanel.svelte': 4,
    'frontend/src/lib/components/panels/CascadingPanel.svelte': 6,
    'frontend/src/lib/components/panels/CodePanel.svelte': 5,
    'frontend/src/lib/components/panels/CompressionSettings.svelte': 5,
    'frontend/src/lib/components/panels/EventTimeline.svelte': 4,
    'frontend/src/lib/components/panels/ExecPipelinePanel.svelte': 16,
    'frontend/src/lib/components/panels/FileManager.svelte': 6,
    'frontend/src/lib/components/panels/HumanizerPanel.svelte': 5,
    'frontend/src/lib/components/panels/LearnedRouterPanel.svelte': 7,
    'frontend/src/lib/components/panels/MemoryPanel.svelte': 9,
    'frontend/src/lib/components/panels/ModelAssignment.svelte': 3,
    'frontend/src/lib/components/panels/ModelProfilePanel.svelte': 12,
    'frontend/src/lib/components/panels/NotesDrawingCanvas.svelte': 3,
    'frontend/src/lib/components/panels/NotesPanel.svelte': 1,
    'frontend/src/lib/components/panels/PanelToggle.svelte': 9,
    'frontend/src/lib/components/panels/PerformanceDashboard.svelte': 4,
    'frontend/src/lib/components/panels/PipelineEditor.svelte': 5,
    'frontend/src/lib/components/panels/PipelinePanel.svelte': 16,
    'frontend/src/lib/components/panels/PipelineStepEditor.svelte': 2,
    'frontend/src/lib/components/panels/PluginsQuickPanel.svelte': 3,
    'frontend/src/lib/components/panels/ProfilerDashboard.svelte': 4,
    'frontend/src/lib/components/panels/ProjectList.svelte': 6,
    'frontend/src/lib/components/panels/PromptConfigPanel.svelte': 9,
    'frontend/src/lib/components/panels/ProxySettingsPanel.svelte': 5,
    'frontend/src/lib/components/panels/SandboxFileManager.svelte': 5,
    'frontend/src/lib/components/panels/SandboxHostExplorer.svelte': 1,
    'frontend/src/lib/components/panels/SandboxUploadZone.svelte': 1,
    'frontend/src/lib/components/panels/SkillsPanel.svelte': 1,
    'frontend/src/lib/components/panels/SpeculativePanel.svelte': 2,
    'frontend/src/lib/components/panels/TelemetryDashboard.svelte': 3,
    'frontend/src/lib/components/panels/TelemetryHistoryPanel.svelte': 15,
    'frontend/src/lib/components/panels/benchmark/BenchmarkHeadToHead.svelte': 1,
    'frontend/src/lib/components/panels/benchmark/BenchmarkLeaderboard.svelte': 1,
    'frontend/src/lib/components/panels/benchmark/BenchmarkProfiles.svelte': 8,
    'frontend/src/lib/components/panels/benchmark/BenchmarkRunSection.svelte': 7,
    'frontend/src/lib/components/panels/benchmark/BenchmarkTrends.svelte': 1,
    'frontend/src/lib/components/rag/BatchUpload.svelte': 5,
    'frontend/src/lib/components/rag/DocumentManager.svelte': 7,
    'frontend/src/lib/components/rag/FolderScan.svelte': 2,
    'frontend/src/lib/components/rag/IngestProgress.svelte': 1,
    'frontend/src/lib/components/settings/AppPasswordsPanel.svelte': 8,
    'frontend/src/lib/components/settings/AuditChainPanel.svelte': 5,
    'frontend/src/lib/components/settings/BackupRestorePanel.svelte': 9,
    'frontend/src/lib/components/settings/ContextOptimizerPanel.svelte': 5,
    'frontend/src/lib/components/settings/FineTunePanel.svelte': 9,
    'frontend/src/lib/components/settings/HardeningPanel.svelte': 1,
    'frontend/src/lib/components/settings/KeyCeremonyPanel.svelte': 11,
    'frontend/src/lib/components/settings/KnowledgeBasePanel.svelte': 6,
    'frontend/src/lib/components/settings/ModelHealthWidget.svelte': 2,
    'frontend/src/lib/components/settings/PerformanceTunerPanel.svelte': 8,
    'frontend/src/lib/components/settings/PluginAllowlistPanel.svelte': 9,
    'frontend/src/lib/components/settings/PluginMarketplace.svelte': 11,
    'frontend/src/lib/components/settings/PluginsPanel.svelte': 9,
    'frontend/src/lib/components/settings/PresetManager.svelte': 9,
    'frontend/src/lib/components/settings/RAGDashboardPanel.svelte': 1,
    'frontend/src/lib/components/settings/RecoveryCodesPanel.svelte': 6,
    'frontend/src/lib/components/settings/RemoteAccessPanel.svelte': 4,
    'frontend/src/lib/components/settings/SearchKillSwitchPanel.svelte': 9,
    'frontend/src/lib/components/settings/SecurityModePanel.svelte': 6,
    'frontend/src/lib/components/settings/ShortcutSettings.svelte': 3,
    'frontend/src/lib/components/settings/SpeculativeDecodingPanel.svelte': 5,
    'frontend/src/lib/components/settings/TOTPSetup.svelte': 6,
    'frontend/src/lib/components/settings/VisionModelSelector.svelte': 2,
    'frontend/src/lib/components/settings/WebAuthnSetup.svelte': 6,
    'frontend/src/lib/components/settings/sections/AppearanceSection.svelte': 4,
    'frontend/src/lib/components/settings/sections/ConversationDefaults.svelte': 1,
    'frontend/src/lib/components/ui/ErrorBoundary.svelte': 1,
    'frontend/src/lib/components/ui/NotificationCenter.svelte': 2,
    'frontend/src/lib/components/ui/OnboardingOverlay.svelte': 1,
    'frontend/src/lib/components/ui/ThemeSwitcher.svelte': 3,
    'frontend/src/lib/components/ui/UserMenu.svelte': 3,
    'frontend/src/routes/(app)/(workshop)/workshop/verify/+page.svelte': 2,
}
UR2_LEDGER = {
    'frontend/src/lib/components/chat/BranchExplorer.svelte': 4,
    'frontend/src/lib/components/chat/ChatControlBar.svelte': 2,
    'frontend/src/lib/components/chat/ChatInput.svelte': 2,
    'frontend/src/lib/components/chat/FeedbackWidget.svelte': 1,
    'frontend/src/lib/components/chat/FileUpload.svelte': 1,
    'frontend/src/lib/components/panels/AnalyticsDashboard.svelte': 1,
    'frontend/src/lib/components/panels/AnswerVerifier.svelte': 2,
    'frontend/src/lib/components/panels/CacheStatsPanel.svelte': 6,
    'frontend/src/lib/components/panels/CascadingPanel.svelte': 9,
    'frontend/src/lib/components/panels/CitationVerifier.svelte': 2,
    'frontend/src/lib/components/panels/CodePanel.svelte': 2,
    'frontend/src/lib/components/panels/CompressionSettings.svelte': 7,
    'frontend/src/lib/components/panels/ExecPipelinePanel.svelte': 5,
    'frontend/src/lib/components/panels/FileManager.svelte': 1,
    'frontend/src/lib/components/panels/HumanizerPanel.svelte': 3,
    'frontend/src/lib/components/panels/LearnedRouterPanel.svelte': 5,
    'frontend/src/lib/components/panels/MemoriesPanel.svelte': 1,
    'frontend/src/lib/components/panels/MemoryPanel.svelte': 2,
    'frontend/src/lib/components/panels/ModelAssignment.svelte': 3,
    'frontend/src/lib/components/panels/ModelProfilePanel.svelte': 3,
    'frontend/src/lib/components/panels/NotesMediaGallery.svelte': 1,
    'frontend/src/lib/components/panels/NotesPanel.svelte': 3,
    'frontend/src/lib/components/panels/PipelineEditor.svelte': 5,
    'frontend/src/lib/components/panels/PipelinePanel.svelte': 11,
    'frontend/src/lib/components/panels/PipelineStepEditor.svelte': 7,
    'frontend/src/lib/components/panels/ProjectDetail.svelte': 1,
    'frontend/src/lib/components/panels/PromptConfigPanel.svelte': 4,
    'frontend/src/lib/components/panels/ProxySettingsPanel.svelte': 5,
    'frontend/src/lib/components/panels/SandboxDiffReview.svelte': 2,
    'frontend/src/lib/components/panels/SandboxFileManager.svelte': 2,
    'frontend/src/lib/components/panels/SandboxUploadZone.svelte': 1,
    'frontend/src/lib/components/panels/SpeculativePanel.svelte': 10,
    'frontend/src/lib/components/panels/SyncPanel.svelte': 2,
    'frontend/src/lib/components/panels/TelemetryHistoryPanel.svelte': 4,
    'frontend/src/lib/components/panels/benchmark/BenchmarkHeadToHead.svelte': 2,
    'frontend/src/lib/components/panels/benchmark/BenchmarkProfiles.svelte': 6,
    'frontend/src/lib/components/panels/benchmark/BenchmarkRunSection.svelte': 5,
    'frontend/src/lib/components/panels/benchmark/BenchmarkTrends.svelte': 1,
    'frontend/src/lib/components/rag/BatchUpload.svelte': 3,
    'frontend/src/lib/components/rag/DocumentManager.svelte': 5,
    'frontend/src/lib/components/rag/FolderScan.svelte': 3,
    'frontend/src/lib/components/settings/AppPasswordsPanel.svelte': 1,
    'frontend/src/lib/components/settings/AuditChainPanel.svelte': 2,
    'frontend/src/lib/components/settings/BackupRestorePanel.svelte': 2,
    'frontend/src/lib/components/settings/ContextOptimizerPanel.svelte': 5,
    'frontend/src/lib/components/settings/FineTunePanel.svelte': 11,
    'frontend/src/lib/components/settings/KeyCeremonyPanel.svelte': 2,
    'frontend/src/lib/components/settings/KnowledgeBasePanel.svelte': 5,
    'frontend/src/lib/components/settings/PerformanceTunerPanel.svelte': 1,
    'frontend/src/lib/components/settings/PluginAllowlistPanel.svelte': 3,
    'frontend/src/lib/components/settings/PluginMarketplace.svelte': 11,
    'frontend/src/lib/components/settings/PluginsPanel.svelte': 3,
    'frontend/src/lib/components/settings/PresetManager.svelte': 14,
    'frontend/src/lib/components/settings/RemoteAccessPanel.svelte': 3,
    'frontend/src/lib/components/settings/SearchKillSwitchPanel.svelte': 5,
    'frontend/src/lib/components/settings/SecurityModePanel.svelte': 3,
    'frontend/src/lib/components/settings/SpeculativeDecodingPanel.svelte': 5,
    'frontend/src/lib/components/settings/TOTPSetup.svelte': 1,
    'frontend/src/lib/components/settings/VisionModelSelector.svelte': 2,
    'frontend/src/lib/components/settings/WebAuthnSetup.svelte': 1,
}
UR3_LEDGER = {
    'frontend/src/lib/components/chat/BranchExplorer.svelte': 10,
    'frontend/src/lib/components/chat/BranchTreeNodeItem.svelte': 3,
    'frontend/src/lib/components/chat/ChatControlBar.svelte': 14,
    'frontend/src/lib/components/chat/ChatInput.svelte': 9,
    'frontend/src/lib/components/chat/ChatMessage.svelte': 12,
    'frontend/src/lib/components/chat/CodingAgentInline.svelte': 15,
    'frontend/src/lib/components/chat/CodingAgentProgress.svelte': 16,
    'frontend/src/lib/components/chat/ContextBar.svelte': 8,
    'frontend/src/lib/components/chat/ContextPanel.svelte': 4,
    'frontend/src/lib/components/chat/FeedbackWidget.svelte': 5,
    'frontend/src/lib/components/chat/FileUpload.svelte': 1,
    'frontend/src/lib/components/chat/MessageSkeleton.svelte': 3,
    'frontend/src/lib/components/chat/ModelSelector.svelte': 2,
    'frontend/src/lib/components/chat/PluginPermissionBadge.svelte': 1,
    'frontend/src/lib/components/chat/ProjectContextBadge.svelte': 5,
    'frontend/src/lib/components/chat/ProjectLinker.svelte': 9,
    'frontend/src/lib/components/chat/ReasoningDisplay.svelte': 1,
    'frontend/src/lib/components/chat/SandboxIsolationBadge.svelte': 1,
    'frontend/src/lib/components/chat/ToolCallApprovalDrawer.svelte': 8,
    'frontend/src/lib/components/health/CacheManager.svelte': 33,
    'frontend/src/lib/components/health/HealthDashboard.svelte': 52,
    'frontend/src/lib/components/panels/AnalyticsDashboard.svelte': 61,
    'frontend/src/lib/components/panels/ArtifactPanel.svelte': 2,
    'frontend/src/lib/components/panels/CacheStatsPanel.svelte': 43,
    'frontend/src/lib/components/panels/CascadingPanel.svelte': 46,
    'frontend/src/lib/components/panels/CodePanel.svelte': 2,
    'frontend/src/lib/components/panels/CompressionSettings.svelte': 55,
    'frontend/src/lib/components/panels/EventTimeline.svelte': 3,
    'frontend/src/lib/components/panels/ExecPipelinePanel.svelte': 48,
    'frontend/src/lib/components/panels/FileManager.svelte': 34,
    'frontend/src/lib/components/panels/HumanizerPanel.svelte': 38,
    'frontend/src/lib/components/panels/LearnedRouterPanel.svelte': 77,
    'frontend/src/lib/components/panels/MemoryPanel.svelte': 1,
    'frontend/src/lib/components/panels/ModelProfilePanel.svelte': 52,
    'frontend/src/lib/components/panels/NotesDrawingCanvas.svelte': 2,
    'frontend/src/lib/components/panels/PanelToggle.svelte': 9,
    'frontend/src/lib/components/panels/PerformanceDashboard.svelte': 50,
    'frontend/src/lib/components/panels/PipelineEditor.svelte': 25,
    'frontend/src/lib/components/panels/PluginsQuickPanel.svelte': 16,
    'frontend/src/lib/components/panels/ProfilerDashboard.svelte': 3,
    'frontend/src/lib/components/panels/ProjectDetail.svelte': 32,
    'frontend/src/lib/components/panels/ProjectList.svelte': 11,
    'frontend/src/lib/components/panels/PromptConfigPanel.svelte': 47,
    'frontend/src/lib/components/panels/ProxySettingsPanel.svelte': 78,
    'frontend/src/lib/components/panels/SpeculativePanel.svelte': 51,
    'frontend/src/lib/components/panels/TelemetryDashboard.svelte': 1,
    'frontend/src/lib/components/panels/TelemetryHistoryPanel.svelte': 4,
    'frontend/src/lib/components/panels/benchmark/BenchmarkCompareSection.svelte': 5,
    'frontend/src/lib/components/panels/benchmark/BenchmarkHistorySection.svelte': 1,
    'frontend/src/lib/components/panels/benchmark/BenchmarkLeaderboard.svelte': 8,
    'frontend/src/lib/components/panels/benchmark/BenchmarkRunDrawer.svelte': 20,
    'frontend/src/lib/components/panels/benchmark/BenchmarkRunSection.svelte': 9,
    'frontend/src/lib/components/panels/benchmark/BenchmarkTrends.svelte': 6,
    'frontend/src/lib/components/rag/BatchUpload.svelte': 21,
    'frontend/src/lib/components/rag/DocumentManager.svelte': 23,
    'frontend/src/lib/components/rag/FolderScan.svelte': 8,
    'frontend/src/lib/components/rag/IngestProgress.svelte': 23,
    'frontend/src/lib/components/settings/AppPasswordsPanel.svelte': 23,
    'frontend/src/lib/components/settings/AuditChainPanel.svelte': 52,
    'frontend/src/lib/components/settings/BackupRestorePanel.svelte': 51,
    'frontend/src/lib/components/settings/ContextOptimizerPanel.svelte': 54,
    'frontend/src/lib/components/settings/FineTunePanel.svelte': 71,
    'frontend/src/lib/components/settings/HardeningPanel.svelte': 56,
    'frontend/src/lib/components/settings/KeyCeremonyPanel.svelte': 62,
    'frontend/src/lib/components/settings/KnowledgeBasePanel.svelte': 35,
    'frontend/src/lib/components/settings/ModelHealthWidget.svelte': 2,
    'frontend/src/lib/components/settings/PerformanceTunerPanel.svelte': 48,
    'frontend/src/lib/components/settings/PluginAllowlistPanel.svelte': 63,
    'frontend/src/lib/components/settings/PluginMarketplace.svelte': 62,
    'frontend/src/lib/components/settings/PluginsPanel.svelte': 51,
    'frontend/src/lib/components/settings/RAGDashboardPanel.svelte': 46,
    'frontend/src/lib/components/settings/RecoveryCodesPanel.svelte': 21,
    'frontend/src/lib/components/settings/RemoteAccessPanel.svelte': 41,
    'frontend/src/lib/components/settings/SearchKillSwitchPanel.svelte': 66,
    'frontend/src/lib/components/settings/SecurityModePanel.svelte': 35,
    'frontend/src/lib/components/settings/SpeculativeDecodingPanel.svelte': 47,
    'frontend/src/lib/components/settings/TOTPSetup.svelte': 30,
    'frontend/src/lib/components/settings/VisionModelSelector.svelte': 16,
    'frontend/src/lib/components/settings/WebAuthnSetup.svelte': 20,
    'frontend/src/lib/components/sidebar/SecurityBadge.svelte': 1,
    'frontend/src/lib/components/ui/ErrorBoundary.svelte': 5,
    'frontend/src/lib/components/ui/FeatureUnavailable.svelte': 6,
    'frontend/src/lib/components/ui/SkeletonLoader.svelte': 8,
    'frontend/src/routes/(app)/(use)/chat/[id]/+page.svelte': 2,
    'frontend/src/routes/(app)/(workshop)/workshop/+page.svelte': 4,
}
UR4_LEDGER = {
    'frontend/src/lib/components/chat/BranchExplorer.svelte': 8,
    'frontend/src/lib/components/chat/BranchTreeNodeItem.svelte': 3,
    'frontend/src/lib/components/chat/ChatInput.svelte': 1,
    'frontend/src/lib/components/chat/ContextBar.svelte': 1,
    'frontend/src/lib/components/chat/CorrectionIndicator.svelte': 8,
    'frontend/src/lib/components/chat/FileUpload.svelte': 3,
    'frontend/src/lib/components/chat/LiveMetricsOverlay.svelte': 5,
    'frontend/src/lib/components/chat/ModelSelector.svelte': 4,
    'frontend/src/lib/components/chat/PresetSelector.svelte': 4,
    'frontend/src/lib/components/chat/ProjectContextBadge.svelte': 2,
    'frontend/src/lib/components/chat/ProjectLinker.svelte': 2,
    'frontend/src/lib/components/chat/ReasoningDisplay.svelte': 2,
    'frontend/src/lib/components/chat/RoutingIndicator.svelte': 5,
    'frontend/src/lib/components/health/CacheManager.svelte': 1,
    'frontend/src/lib/components/health/HealthDashboard.svelte': 4,
    'frontend/src/lib/components/panels/CompressionSettings.svelte': 7,
    'frontend/src/lib/components/panels/EventTimeline.svelte': 5,
    'frontend/src/lib/components/panels/FileManager.svelte': 9,
    'frontend/src/lib/components/panels/MemoryPanel.svelte': 10,
    'frontend/src/lib/components/panels/ModelAssignment.svelte': 9,
    'frontend/src/lib/components/panels/ModelProfilePanel.svelte': 1,
    'frontend/src/lib/components/panels/PluginsQuickPanel.svelte': 2,
    'frontend/src/lib/components/panels/ProfilerDashboard.svelte': 4,
    'frontend/src/lib/components/panels/ProjectDetail.svelte': 6,
    'frontend/src/lib/components/panels/ProjectList.svelte': 2,
    'frontend/src/lib/components/panels/PromptConfigPanel.svelte': 16,
    'frontend/src/lib/components/panels/ProxySettingsPanel.svelte': 2,
    'frontend/src/lib/components/panels/SandboxDiffReview.svelte': 4,
    'frontend/src/lib/components/panels/SandboxFileManager.svelte': 5,
    'frontend/src/lib/components/panels/SandboxHostExplorer.svelte': 1,
    'frontend/src/lib/components/panels/SandboxSettingsStrip.svelte': 1,
    'frontend/src/lib/components/panels/TelemetryHistoryPanel.svelte': 4,
    'frontend/src/lib/components/panels/benchmark/benchmark.css': 18,
    'frontend/src/lib/components/settings/ContextOptimizerPanel.svelte': 4,
    'frontend/src/lib/components/settings/ModelHealthWidget.svelte': 5,
    'frontend/src/lib/components/settings/PerformanceTunerPanel.svelte': 6,
    'frontend/src/lib/components/settings/PluginsPanel.svelte': 2,
    'frontend/src/lib/components/settings/ShortcutSettings.svelte': 3,
    'frontend/src/lib/components/settings/SpeculativeDecodingPanel.svelte': 5,
}
UR5_LEDGER = {
    'frontend/src/lib/components/chat/FileUpload.svelte': 1,
    'frontend/src/lib/components/chat/LiveMetricsOverlay.svelte': 1,
    'frontend/src/lib/components/panels/EventTimeline.svelte': 1,
    'frontend/src/lib/components/panels/ModelAssignment.svelte': 1,
    'frontend/src/lib/components/panels/ProfilerDashboard.svelte': 2,
    'frontend/src/lib/components/panels/RemoteChannelPanel.svelte': 1,
    'frontend/src/lib/components/panels/SandboxFileManager.svelte': 18,
    'frontend/src/lib/components/panels/TelemetryDashboard.svelte': 2,
    'frontend/src/lib/components/panels/TelemetryHistoryPanel.svelte': 8,
    'frontend/src/lib/components/settings/HardeningPanel.svelte': 5,
    'frontend/src/lib/components/settings/RecoveryCodesPanel.svelte': 1,
    'frontend/src/lib/components/settings/RemoteAccessPanel.svelte': 3,
    'frontend/src/lib/components/settings/TOTPSetup.svelte': 2,
    'frontend/src/lib/components/settings/WebAuthnSetup.svelte': 1,
}
UR6_LEDGER = {
    'frontend/src/lib/components/chat/ChatInput.svelte': 1,
    'frontend/src/lib/components/chat/ChatMessage.svelte': 1,
    'frontend/src/lib/components/panels/ArtifactPanel.svelte': 1,
    'frontend/src/lib/components/panels/ExecPipelinePanel.svelte': 1,
    'frontend/src/lib/components/panels/FileManager.svelte': 1,
    'frontend/src/lib/components/panels/MemoryPanel.svelte': 1,
    'frontend/src/lib/components/panels/PipelinePanel.svelte': 1,
}
UR7_LEDGER = {
    'frontend/src/lib/components/panels/benchmark/BenchmarkHeadToHead.svelte': 1,
    'frontend/src/lib/components/panels/benchmark/BenchmarkRunSection.svelte': 1,
    'frontend/src/lib/components/panels/benchmark/BenchmarkTrends.svelte': 1,
    'frontend/src/lib/components/settings/SpeculativeDecodingPanel.svelte': 1,
}
UR8_LEDGER = {
    'frontend/src/lib/api/artifacts.ts': 2,
    'frontend/src/lib/api/cache.ts': 1,
    'frontend/src/lib/api/chat.ts': 1,
    'frontend/src/lib/api/code.ts': 2,
    'frontend/src/lib/api/context.ts': 1,
    'frontend/src/lib/api/export.ts': 1,
    'frontend/src/lib/api/files.ts': 3,
    'frontend/src/lib/api/memory.ts': 2,
    'frontend/src/lib/api/pipelines.ts': 4,
    'frontend/src/lib/api/presets.ts': 2,
    'frontend/src/lib/api/settings.ts': 2,
    'frontend/src/lib/components/chat/CorrectionIndicator.svelte': 1,
    'frontend/src/lib/components/panels/ArtifactPanel.svelte': 2,
    'frontend/src/lib/components/panels/CodePanel.svelte': 1,
    'frontend/src/lib/components/panels/MemoryPanel.svelte': 1,
    'frontend/src/lib/components/panels/PanelToggle.svelte': 2,
    'frontend/src/lib/components/panels/PipelineEditor.svelte': 2,
    'frontend/src/lib/components/settings/PresetManager.svelte': 1,
    'frontend/src/lib/stores/chat.ts': 18,
    'frontend/src/lib/stores/conversations.ts': 3,
    'frontend/src/lib/stores/panels.ts': 4,
    'frontend/src/lib/types.ts': 1,
    'frontend/src/routes/(app)/(use)/chat/[id]/+page.svelte': 1,
}
UR9_LEDGER = {
    'frontend/src/lib/components/chat/ContextPanel.svelte': 1,
    'frontend/src/lib/components/chat/LiveMetricsOverlay.svelte': 1,
    'frontend/src/lib/components/chat/ToolCallApprovalDrawer.svelte': 1,
    'frontend/src/lib/components/panels/AgentPanel.svelte': 1,
    'frontend/src/lib/components/panels/EventTimeline.svelte': 1,
    'frontend/src/lib/components/panels/PerformanceDashboard.svelte': 1,
    'frontend/src/lib/components/panels/ProfilerDashboard.svelte': 1,
    'frontend/src/lib/components/panels/TelemetryDashboard.svelte': 1,
    'frontend/src/lib/components/panels/TelemetryHistoryPanel.svelte': 1,
    'frontend/src/lib/components/panels/benchmark/BenchmarkRunSection.svelte': 1,
    'frontend/src/lib/components/rag/IngestProgress.svelte': 1,
    'frontend/src/lib/components/settings/ModelHealthWidget.svelte': 1,
    'frontend/src/lib/components/settings/PerformanceTunerPanel.svelte': 1,
    'frontend/src/lib/components/settings/SearchKillSwitchPanel.svelte': 1,
    'frontend/src/lib/components/settings/SecurityModePanel.svelte': 1,
    'frontend/src/lib/components/settings/SpeculativeDecodingPanel.svelte': 1,
}
UR10_LEDGER = {
    'frontend/src/lib/components/chat/CorrectionIndicator.svelte': 2,
    'frontend/src/lib/components/panels/CascadingPanel.svelte': 1,
    'frontend/src/lib/components/panels/EventTimeline.svelte': 1,
    'frontend/src/lib/components/panels/SandboxFileManager.svelte': 1,
    'frontend/src/lib/components/rag/BatchUpload.svelte': 1,
    'frontend/src/lib/components/rag/DocumentManager.svelte': 1,
    'frontend/src/lib/components/rag/IngestProgress.svelte': 2,
    'frontend/src/lib/components/settings/AuditChainPanel.svelte': 4,
    'frontend/src/lib/components/settings/KeyCeremonyPanel.svelte': 1,
    'frontend/src/lib/components/settings/KnowledgeBasePanel.svelte': 1,
    'frontend/src/lib/components/settings/PluginMarketplace.svelte': 4,
    'frontend/src/lib/components/settings/PresetManager.svelte': 1,
}
UR11_LEDGER = {
    'frontend/src/lib/components/chat/BranchExplorer.svelte': 10,
    'frontend/src/lib/components/chat/BranchTreeNodeItem.svelte': 2,
    'frontend/src/lib/components/chat/ChatControlBar.svelte': 3,
    'frontend/src/lib/components/chat/ChatInput.svelte': 1,
    'frontend/src/lib/components/chat/ChatMessage.svelte': 6,
    'frontend/src/lib/components/chat/CodingAgentInline.svelte': 3,
    'frontend/src/lib/components/chat/CodingAgentProgress.svelte': 4,
    'frontend/src/lib/components/chat/ContextBar.svelte': 5,
    'frontend/src/lib/components/chat/CorrectionIndicator.svelte': 2,
    'frontend/src/lib/components/chat/ExportDialog.svelte': 1,
    'frontend/src/lib/components/chat/FeedbackWidget.svelte': 1,
    'frontend/src/lib/components/chat/LiveMetricsOverlay.svelte': 1,
    'frontend/src/lib/components/chat/ModelSelector.svelte': 2,
    'frontend/src/lib/components/chat/ProjectLinker.svelte': 3,
    'frontend/src/lib/components/chat/RoutingIndicator.svelte': 4,
    'frontend/src/lib/components/health/CacheManager.svelte': 1,
    'frontend/src/lib/components/health/HealthDashboard.svelte': 3,
    'frontend/src/lib/components/layout/StatusFooter.svelte': 2,
    'frontend/src/lib/components/panels/AnalyticsDashboard.svelte': 14,
    'frontend/src/lib/components/panels/AnswerVerifier.svelte': 2,
    'frontend/src/lib/components/panels/ArtifactPanel.svelte': 1,
    'frontend/src/lib/components/panels/CacheStatsPanel.svelte': 7,
    'frontend/src/lib/components/panels/CascadingPanel.svelte': 6,
    'frontend/src/lib/components/panels/CitationVerifier.svelte': 2,
    'frontend/src/lib/components/panels/CodePanel.svelte': 1,
    'frontend/src/lib/components/panels/CompressionSettings.svelte': 8,
    'frontend/src/lib/components/panels/EventTimeline.svelte': 5,
    'frontend/src/lib/components/panels/ExecPipelinePanel.svelte': 2,
    'frontend/src/lib/components/panels/FileManager.svelte': 7,
    'frontend/src/lib/components/panels/GovernorPanel.svelte': 1,
    'frontend/src/lib/components/panels/HumanizerPanel.svelte': 5,
    'frontend/src/lib/components/panels/LearnedRouterPanel.svelte': 7,
    'frontend/src/lib/components/panels/MemoriesPanel.svelte': 1,
    'frontend/src/lib/components/panels/MemoryPanel.svelte': 1,
    'frontend/src/lib/components/panels/ModelAssignment.svelte': 6,
    'frontend/src/lib/components/panels/ModelProfilePanel.svelte': 6,
    'frontend/src/lib/components/panels/NoteActionPanel.svelte': 1,
    'frontend/src/lib/components/panels/NotesDrawingCanvas.svelte': 4,
    'frontend/src/lib/components/panels/NotesMediaGallery.svelte': 2,
    'frontend/src/lib/components/panels/NotesPanel.svelte': 1,
    'frontend/src/lib/components/panels/NotesVoiceCapture.svelte': 2,
    'frontend/src/lib/components/panels/PerformanceDashboard.svelte': 9,
    'frontend/src/lib/components/panels/PipelineEditor.svelte': 3,
    'frontend/src/lib/components/panels/PluginsQuickPanel.svelte': 4,
    'frontend/src/lib/components/panels/ProfilerDashboard.svelte': 8,
    'frontend/src/lib/components/panels/ProjectDetail.svelte': 5,
    'frontend/src/lib/components/panels/ProjectList.svelte': 5,
    'frontend/src/lib/components/panels/PromptConfigPanel.svelte': 10,
    'frontend/src/lib/components/panels/ProxySettingsPanel.svelte': 14,
    'frontend/src/lib/components/panels/RemoteChannelPanel.svelte': 1,
    'frontend/src/lib/components/panels/SandboxDiffReview.svelte': 2,
    'frontend/src/lib/components/panels/SandboxFileManager.svelte': 6,
    'frontend/src/lib/components/panels/SandboxHostExplorer.svelte': 2,
    'frontend/src/lib/components/panels/SandboxSettingsStrip.svelte': 1,
    'frontend/src/lib/components/panels/SandboxUploadZone.svelte': 1,
    'frontend/src/lib/components/panels/SandboxWorkspaceList.svelte': 1,
    'frontend/src/lib/components/panels/SpeculativePanel.svelte': 14,
    'frontend/src/lib/components/panels/SyncPanel.svelte': 4,
    'frontend/src/lib/components/panels/TelemetryDashboard.svelte': 4,
    'frontend/src/lib/components/panels/TelemetryHistoryPanel.svelte': 8,
    'frontend/src/lib/components/panels/VerdictHistory.svelte': 1,
    'frontend/src/lib/components/panels/benchmark/BenchmarkRunDrawer.svelte': 1,
    'frontend/src/lib/components/panels/benchmark/BenchmarkRunSection.svelte': 2,
    'frontend/src/lib/components/panels/benchmark/benchmark.css': 21,
    'frontend/src/lib/components/rag/BatchUpload.svelte': 4,
    'frontend/src/lib/components/rag/DocumentManager.svelte': 7,
    'frontend/src/lib/components/rag/FolderScan.svelte': 1,
    'frontend/src/lib/components/rag/IngestProgress.svelte': 2,
    'frontend/src/lib/components/settings/AppPasswordsPanel.svelte': 7,
    'frontend/src/lib/components/settings/AuditChainPanel.svelte': 10,
    'frontend/src/lib/components/settings/BackupRestorePanel.svelte': 14,
    'frontend/src/lib/components/settings/ContextOptimizerPanel.svelte': 5,
    'frontend/src/lib/components/settings/FineTunePanel.svelte': 20,
    'frontend/src/lib/components/settings/HardeningPanel.svelte': 5,
    'frontend/src/lib/components/settings/KeyCeremonyPanel.svelte': 8,
    'frontend/src/lib/components/settings/KnowledgeBasePanel.svelte': 6,
    'frontend/src/lib/components/settings/ModelHealthWidget.svelte': 3,
    'frontend/src/lib/components/settings/PerformanceTunerPanel.svelte': 7,
    'frontend/src/lib/components/settings/PluginAllowlistPanel.svelte': 14,
    'frontend/src/lib/components/settings/PluginMarketplace.svelte': 10,
    'frontend/src/lib/components/settings/PluginsPanel.svelte': 9,
    'frontend/src/lib/components/settings/RAGDashboardPanel.svelte': 13,
    'frontend/src/lib/components/settings/RecoveryCodesPanel.svelte': 5,
    'frontend/src/lib/components/settings/RemoteAccessPanel.svelte': 9,
    'frontend/src/lib/components/settings/SearchKillSwitchPanel.svelte': 13,
    'frontend/src/lib/components/settings/SecurityModePanel.svelte': 10,
    'frontend/src/lib/components/settings/ShortcutSettings.svelte': 6,
    'frontend/src/lib/components/settings/SpeculativeDecodingPanel.svelte': 6,
    'frontend/src/lib/components/settings/TOTPSetup.svelte': 8,
    'frontend/src/lib/components/settings/VisionModelSelector.svelte': 4,
    'frontend/src/lib/components/settings/WebAuthnSetup.svelte': 5,
    'frontend/src/lib/components/settings/sections/AppearanceSection.svelte': 6,
    'frontend/src/lib/components/settings/sections/ConversationDefaults.svelte': 2,
    'frontend/src/lib/components/ui/ErrorBoundary.svelte': 1,
    'frontend/src/lib/components/ui/FeatureUnavailable.svelte': 1,
    'frontend/src/lib/components/ui/SkeletonLoader.svelte': 2,
    'frontend/src/routes/(app)/(use)/chat/[id]/+page.svelte': 3,
    'frontend/src/routes/(app)/(workshop)/workshop/+page.svelte': 1,
    'frontend/src/routes/(app)/(workshop)/workshop/verify/+page.svelte': 1,
    'frontend/src/styles/theme.css': 5,
    'frontend/src/styles/transitions.css': 1,
}
UR12_LEDGER = {
    'frontend/src/lib/api/keyCeremony.ts': 1,
    'frontend/src/lib/components/chat/BranchExplorer.svelte': 3,
    'frontend/src/lib/components/chat/ContextPanel.svelte': 2,
    'frontend/src/lib/components/chat/CorrectionIndicator.svelte': 4,
    'frontend/src/lib/components/chat/LiveMetricsOverlay.svelte': 2,
    'frontend/src/lib/components/chat/ModelSelector.svelte': 2,
    'frontend/src/lib/components/chat/RoutingIndicator.svelte': 2,
    'frontend/src/lib/components/layout/StatusFooter.svelte': 2,
    'frontend/src/lib/components/panels/AgentPanel.svelte': 2,
    'frontend/src/lib/components/panels/AnswerVerifier.svelte': 2,
    'frontend/src/lib/components/panels/CitationVerifier.svelte': 2,
    'frontend/src/lib/components/panels/MemoriesPanel.svelte': 2,
    'frontend/src/lib/components/panels/ModelAssignment.svelte': 4,
    'frontend/src/lib/components/panels/NoteActionPanel.svelte': 2,
    'frontend/src/lib/components/panels/ProfilerDashboard.svelte': 4,
    'frontend/src/lib/components/panels/SandboxDiffReview.svelte': 1,
    'frontend/src/lib/components/panels/SandboxFileManager.svelte': 1,
    'frontend/src/lib/components/panels/SyncPanel.svelte': 1,
    'frontend/src/lib/components/panels/VerdictHistory.svelte': 2,
    'frontend/src/lib/components/panels/benchmark/benchmark.css': 2,
    'frontend/src/lib/components/settings/SearchKillSwitchPanel.svelte': 2,
    'frontend/src/lib/components/settings/SecurityModePanel.svelte': 4,
    'frontend/src/lib/components/settings/ShortcutSettings.svelte': 2,
    'frontend/src/lib/components/settings/TOTPSetup.svelte': 1,
    'frontend/src/lib/components/settings/sections/AppearanceSection.svelte': 2,
}
UR13_LEDGER = {
    'frontend/src/lib/components/chat/BranchExplorer.svelte': 3,
    'frontend/src/lib/components/chat/BranchTreeNodeItem.svelte': 1,
    'frontend/src/lib/components/chat/ChatMessage.svelte': 1,
    'frontend/src/lib/components/chat/CodingAgentInline.svelte': 4,
    'frontend/src/lib/components/chat/CodingAgentProgress.svelte': 3,
    'frontend/src/lib/components/chat/ContextBar.svelte': 1,
    'frontend/src/lib/components/chat/ContextPanel.svelte': 2,
    'frontend/src/lib/components/chat/CorrectionIndicator.svelte': 1,
    'frontend/src/lib/components/chat/ModelSelector.svelte': 1,
    'frontend/src/lib/components/chat/ProjectLinker.svelte': 1,
    'frontend/src/lib/components/chat/ReasoningDisplay.svelte': 2,
    'frontend/src/lib/components/chat/RoutingIndicator.svelte': 1,
    'frontend/src/lib/components/chat/ToolCallDisplay.svelte': 1,
    'frontend/src/lib/components/health/HealthDashboard.svelte': 2,
    'frontend/src/lib/components/layout/StatusFooter.svelte': 1,
    'frontend/src/lib/components/panels/AnalyticsDashboard.svelte': 1,
    'frontend/src/lib/components/panels/ArtifactPanel.svelte': 3,
    'frontend/src/lib/components/panels/CascadingPanel.svelte': 2,
    'frontend/src/lib/components/panels/CodePanel.svelte': 11,
    'frontend/src/lib/components/panels/EventTimeline.svelte': 1,
    'frontend/src/lib/components/panels/ExecPipelinePanel.svelte': 1,
    'frontend/src/lib/components/panels/HumanizerPanel.svelte': 1,
    'frontend/src/lib/components/panels/LearnedRouterPanel.svelte': 4,
    'frontend/src/lib/components/panels/MemoryPanel.svelte': 3,
    'frontend/src/lib/components/panels/NoteActionPanel.svelte': 1,
    'frontend/src/lib/components/panels/NotesVoiceCapture.svelte': 1,
    'frontend/src/lib/components/panels/PipelineEditor.svelte': 1,
    'frontend/src/lib/components/panels/PipelinePanel.svelte': 2,
    'frontend/src/lib/components/panels/PipelineStepEditor.svelte': 1,
    'frontend/src/lib/components/panels/PluginsQuickPanel.svelte': 2,
    'frontend/src/lib/components/panels/ProfilerDashboard.svelte': 3,
    'frontend/src/lib/components/panels/ProjectDetail.svelte': 1,
    'frontend/src/lib/components/panels/ProxySettingsPanel.svelte': 2,
    'frontend/src/lib/components/panels/SandboxDiffReview.svelte': 1,
    'frontend/src/lib/components/panels/SandboxFileManager.svelte': 3,
    'frontend/src/lib/components/panels/SandboxHostExplorer.svelte': 1,
    'frontend/src/lib/components/panels/SandboxSettingsStrip.svelte': 1,
    'frontend/src/lib/components/panels/SpeculativePanel.svelte': 2,
    'frontend/src/lib/components/panels/TelemetryHistoryPanel.svelte': 2,
    'frontend/src/lib/components/panels/benchmark/benchmark.css': 6,
    'frontend/src/lib/components/rag/BatchUpload.svelte': 1,
    'frontend/src/lib/components/rag/DocumentManager.svelte': 1,
    'frontend/src/lib/components/rag/IngestProgress.svelte': 1,
    'frontend/src/lib/components/settings/AuditChainPanel.svelte': 2,
    'frontend/src/lib/components/settings/BackupRestorePanel.svelte': 2,
    'frontend/src/lib/components/settings/ContextOptimizerPanel.svelte': 1,
    'frontend/src/lib/components/settings/FineTunePanel.svelte': 1,
    'frontend/src/lib/components/settings/KnowledgeBasePanel.svelte': 2,
    'frontend/src/lib/components/settings/PerformanceTunerPanel.svelte': 3,
    'frontend/src/lib/components/settings/PluginAllowlistPanel.svelte': 1,
    'frontend/src/lib/components/settings/PluginMarketplace.svelte': 3,
    'frontend/src/lib/components/settings/PluginsPanel.svelte': 2,
    'frontend/src/lib/components/settings/PresetManager.svelte': 1,
    'frontend/src/lib/components/settings/RemoteAccessPanel.svelte': 1,
    'frontend/src/lib/components/settings/SecurityModePanel.svelte': 1,
    'frontend/src/lib/components/settings/ShortcutSettings.svelte': 2,
    'frontend/src/lib/components/settings/SpeculativeDecodingPanel.svelte': 1,
    'frontend/src/routes/(app)/(use)/chat/[id]/+page.svelte': 3,
    'frontend/src/routes/(app)/(workshop)/workshop/+page.svelte': 1,
}
UR14_LEDGER = {
    'frontend/src/lib/components/ui/SkeletonLoader.svelte': 1,
}
UR15_LEDGER = {}
UR16_LEDGER = {
    'frontend/src/lib/api/artifacts.ts': 2,
    'frontend/src/lib/api/attachments.ts': 1,
    'frontend/src/lib/api/auditChain.ts': 1,
    'frontend/src/lib/api/benchmarkV2.ts': 27,
    'frontend/src/lib/api/files.ts': 2,
    'frontend/src/lib/api/modelLifecycle.ts': 12,
    'frontend/src/lib/api/projects.ts': 1,
    'frontend/src/lib/api/rag.ts': 2,
    'frontend/src/lib/api/speculative.ts': 3,
    'frontend/src/lib/api/speculativeDecoding.ts': 5,
    'frontend/src/lib/api/tuner.ts': 9,
}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
