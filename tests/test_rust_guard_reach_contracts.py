#!/usr/bin/env python3
"""Contracts for the guards' reach into the native crates under ``rust/``.

The native crates ship with the rest of the tree, and three guards held
every other published tree to the English-only and nomenclature-free rules
while none of them opened a Rust file: the clean guard and the comment-only
guard named five trees and ``rust/`` was not one of them, and the language
guard read Python alone. These contracts pin the widened reach, and pin the
one part of it that is not a line in a list: a comment model for Rust that
never reads code as comment and refuses what it cannot close.

  * RU1 -- the clean guard's perimeter names ``rust/``, every entry is a
    directory of the tree, the diff it asks git for holds ``rust/`` in its
    pathspec, and a code on a ``///`` or ``//!`` line is charged.
  * RU2 -- the non-ASCII and nomenclature recipes read ``rust/`` through
    that same perimeter, and only the files the manifest lists.
  * RU3 -- the comment-only guard covers exactly the clean guard's trees,
    ``rust/`` among them, and asks git for exactly those.
  * RU4 -- Rust comments are comments: a purge that leaves a line, an outer
    doc, an inner doc and a nested block moves no shape, while a code change
    on the line whose trailing comment lost its code is not accepted.
  * RU5 -- the Rust model never reads code as comment: a code change hidden
    after a nested block, after a raw string holding an odd quote, or after
    a lifetime and an apostrophe inside a string, each beside a purged
    comment, is not accepted; a real comment after a lifetime is stripped.
  * RU6 -- an unterminated block comment (nested), string, raw string or
    char is refused by the comment-only guard and reported unparsable by the
    language guard, never read as clean.
  * RU7 -- the language guard reads ``rust/`` for ``.rs`` files and the
    Python trees for ``.py`` files, skips a Cargo build directory and
    nothing else, and names ``rust/`` unreadable when its reader cannot be
    loaded.
  * RU8 -- Rust prose is read: a ``///`` run, a ``//!`` run, a ``//`` line,
    a nested block and a ``#[doc]`` string are one span each.
  * RU9 -- strings are data: the same prose inside every kind of Rust
    string, after a char holding a quote and after a lifetime, is not read.
  * RU10 -- the added lines of a ``.rs`` file are read, and only those.
  * RU11 -- a comment removed from between two tokens leaves them two: a
    purge that fuses ``pub`` and ``fn`` into one name is not accepted, while
    a purge inside the same comment is.
  * RU12 -- an identifier glued to a literal is its suffix, as the compiler
    reads it, never the prefix of a raw string: a code change behind five
    such literals is not accepted, and a comment after them is read.
  * RU13 -- the source starts where the compiler starts: past one
    byte-order mark, and past a shebang line unless the next token that is
    not the compiler's whitespace or a plain comment is ``[``; a quote on a
    shebang line, or in a comment before that ``[``, hides no change.
  * RU14 -- only a CR LF pair ends a Rust line: both guards read a lone CR
    as a character of its comment, on the base side and the working tree.
  * RU15 -- a ``doc`` value is prose inside an attribute, a ``concat!`` of
    strings included, and a string named ``doc`` anywhere else is data.
  * RU16 -- a TOML file holding a multi-line string is refused, because the
    hash model would read the lines of the value as comments.
  * RU17 -- every tracked file under ``rust/`` is printable ASCII, tab and
    newline: the bytes the guards' Rust reading was measured on, and the
    ASCII rule held in every crate rather than in one source directory.
  * RU18 -- the clean guard's entry point fails a run that adds a code to a
    Rust file, over a real repository.
  * RU19 -- the language guard's entry point fails a run that adds French
    to a Rust file.
  * RU20 -- the language guard's entry point charges standing French in a
    Rust file to the census, and a clean run names the Rust tree it read.
  * RU21 -- the comment-only guard's entry point examines a Rust purge and
    proves it, reading both sides of the diff as the compiler does.
  * RU22 -- the diff readers of the clean and language guards read every
    added line whole: a line is cut at a newline alone, never at a lone CR,
    a vertical tab or a form feed, and a line opening with ``++ `` inside a
    hunk is an added line, never the next file's header.
  * RU23 -- a shebang line holding a lone quote, after a byte-order mark or
    before a character the compiler does not skip, is skipped as the
    compiler skips it: the file is neither refused nor reported unparsable.

Every input carrying nomenclature is assembled from fragments at runtime,
and every French input lives in a string literal, so neither guard trips on
a scan of this file. The entry-point contracts build a repository in a
temporary directory and hand the guards the tree its index was written as:
nothing is committed, and no git variable of the caller's reaches it.

Local-only. Runs under pytest or via the __main__ runner. The guard scripts
live under .github/, outside the importable package, and are loaded through
the shared isolation window; the recipe module is imported as its own suite
imports it.
"""

import contextlib
import importlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate  # noqa: E402

_GUARDS = REPO / ".github" / "scripts"
_CLEAN = "_public_clean_guard_under_rust_reach"
_COMMENT = "_comment_only_guard_under_rust_reach"
_LANGUAGE = "_public_language_guard_under_rust_reach"
_RECIPES = REPO / "scripts" / "surface_recipes.py"

# Fragments assembled at runtime; no clear nomenclature in the source text.
_S = "S"
_CODE = _S + "312"

_RS = "rust/c/src/lib.rs"

# French prose, unaccented, long enough to have a grammar.
_FRENCH = "Retourne la valeur du fichier dans la liste"
_ENGLISH = "Returns the value of the file in the list"


def _load(*names):
    """Load the named guards through one shared window; (modules, restore)."""
    paths = {
        _CLEAN: _GUARDS / "public_clean_guard.py",
        _COMMENT: _GUARDS / "comment_only_guard.py",
        _LANGUAGE: _GUARDS / "public_language_guard.py",
    }
    loaded, restore = isolate(targets={name: paths[name] for name in names})
    return [loaded[name] for name in names], restore


def _write(root, files):
    """Write ``{relative path: text}`` under ``root``."""
    for relative, text in files.items():
        path = Path(root) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _pathspec(argv):
    """The pathspec a git command was handed: what follows ``--``."""
    return argv[argv.index("--") + 1:]


def _capture_git(module):
    """Replace the module's subprocess with one that records argv."""
    seen = []

    def run(cmd, **kwargs):
        seen.append(list(cmd))
        empty = "" if kwargs.get("text") else b""
        return types.SimpleNamespace(stdout=empty, stderr=empty, returncode=0)

    module.subprocess = types.SimpleNamespace(run=run)
    return seen


# ---------------------------------------------------------------------------
# RU1 -- the clean guard reads rust/
# ---------------------------------------------------------------------------
def test_ru1_the_clean_guard_perimeter_holds_rust_and_charges_doc_lines():
    (clean,), restore = _load(_CLEAN)
    try:
        assert "rust/" in clean._SCAN_PATHS, (
            "the native crates ship; leaving their tree out of the perimeter "
            "is a choice, never a limit of a regex over added lines"
        )
        for tree in clean._SCAN_PATHS:
            assert tree.endswith("/") and (REPO / tree.rstrip("/")).is_dir(), (
                f"perimeter entry {tree!r} names no directory in the tree"
            )
        seen = _capture_git(clean)
        assert clean._added_lines_with_paths("HEAD") == []
        assert len(seen) == 1 and seen[0][:2] == ["git", "diff"], seen
        assert _pathspec(seen[0]) == list(clean._SCAN_PATHS), (
            "the diff git is asked for is the perimeter, rust/ included"
        )
        for line in (
            "/// Seeded as in " + _CODE + ", see the notes.",
            "//! Carried since " + _CODE.lower() + " without change.",
        ):
            assert clean.find_violations([line]), (
                f"a code on a Rust doc line is charged: {line!r}"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RU2 -- recipes 3 and 4 read rust/ through the guard's perimeter
# ---------------------------------------------------------------------------
def _recipes():
    """Import the recipe module as its own suite does, path removed after."""
    directory = str(_RECIPES.parent)
    sys.path.insert(0, directory)
    try:
        return importlib.import_module(_RECIPES.stem)
    finally:
        if directory in sys.path:
            sys.path.remove(directory)


def test_ru2_the_recipes_read_rust_through_the_perimeter():
    recipes = _recipes()
    text = "// note " + _CODE + "\nconst NAME: &str = \"caf" + chr(0xE9) + "\";\n"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, {"rust/c/src/x.rs": text, "rust/c/target/y.rs": text})
        (root / "manifest.md5").write_text(
            "0" * 32 + "  rust/c/src/x.rs\n", encoding="utf-8",
        )
        perimeter = recipes.read_perimeter(root / "manifest.md5")
        assert recipes.non_ascii_census(root, perimeter) == (1, 1), (
            "the listed Rust file is charged its one non-ASCII character, and "
            "the unlisted build output is not"
        )
        assert recipes.nomenclature_census(root, perimeter) == (1, 1), (
            "the listed Rust file is charged its one code, and the unlisted "
            "build output is not"
        )
    assert str(_GUARDS) not in sys.path


# ---------------------------------------------------------------------------
# RU3 -- the comment-only guard covers the clean guard's trees exactly
# ---------------------------------------------------------------------------
def test_ru3_the_comment_only_perimeter_is_the_clean_guard_perimeter():
    (clean, comment), restore = _load(_CLEAN, _COMMENT)
    try:
        assert "rust/" in comment._SCAN_PATHS
        assert tuple(comment._SCAN_PATHS) == tuple(clean._SCAN_PATHS), (
            "the two guards cover the same published trees; a list restated "
            "in the second is held equal to the first by this clause"
        )
        seen = _capture_git(comment)
        assert comment._changed_paths("HEAD") == []
        assert len(seen) == 1 and seen[0][:2] == ["git", "diff"], seen
        assert _pathspec(seen[0]) == list(comment._SCAN_PATHS)
    finally:
        restore()


# ---------------------------------------------------------------------------
# RU4 -- Rust comments are comments, and a trailing comment is not a line
# ---------------------------------------------------------------------------
def test_ru4_rust_comments_are_comments_and_code_beside_them_is_not():
    (comment,), restore = _load(_COMMENT)
    try:
        def source(mark):
            return (
                "//! Engine notes" + mark + ".\n"
                "/// Seeds the world" + mark + ".\n"
                "pub fn seed(x: u32) -> u32 {\n"
                "    // tuned" + mark + "\n"
                "    /* outer /* inner */" + mark + " tail */\n"
                "    x + 1\n"
                "}\n"
            )

        before, after = source(" " + _CODE), source("")
        assert comment.debt_count(before) > comment.debt_count(after), (
            "the fixture is a purge the guard examines"
        )
        assert comment.shape(_RS, before) == comment.shape(_RS, after), (
            "line, outer doc, inner doc and nested block comments are "
            "comments: purging them moves no shape"
        )
        assert comment.verdict(_RS, before, after) is None

        before = "fn f() -> u32 {\n    let x = 1; // " + _CODE + "\n    x\n}\n"
        after = "fn f() -> u32 {\n    let x = 2;\n    x\n}\n"
        assert comment.verdict(_RS, before, after) is comment.CANNOT_JUDGE, (
            "the line lost its code in a comment and changed its code: a "
            "line purge must compare the code, not the whole line"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RU5 -- the Rust model never reads code as comment
# ---------------------------------------------------------------------------
_HIDDEN = (
    # A nested block holding a line-comment marker.
    ("/* a /* b */ c // d */ let k = 7;",
     "/* a /* b */ c // d */ let k = 8;"),
    # A raw string holding an odd quote and a line-comment marker.
    ('let s = r#"a " b // c"#; let z = 3; // real',
     'let s = r#"a " b // c"#; let z = 4; // real'),
    # Lifetimes, then an apostrophe inside a string holding the marker.
    ("fn g<'a>(s: &'a str) -> (&'static str, u32) { (\"it's // here\", 3) }",
     "fn g<'a>(s: &'a str) -> (&'static str, u32) { (\"it's // here\", 4) }"),
)


def test_ru5_the_rust_model_never_reads_code_as_comment():
    (comment,), restore = _load(_COMMENT)
    try:
        for hidden_before, hidden_after in _HIDDEN:
            assert comment.comment_free(_RS, hidden_before) != (
                comment.comment_free(_RS, hidden_after)
            ), f"the changed code is code, not comment: {hidden_before!r}"
            before = "// see " + _CODE + "\n" + hidden_before + "\n"
            after = "// see\n" + hidden_after + "\n"
            assert comment.debt_count(before) > comment.debt_count(after)
            # Both sides lex: a refusal for a source the model cannot read
            # would pass the clause below for the wrong reason.
            comment.shape(_RS, before)
            comment.shape(_RS, after)
            # Not accepted is the whole property: an acceptance is None, and
            # whether the guard refuses or does not judge, it has not waved
            # the hidden change through.
            assert comment.verdict(_RS, before, after) is not None, (
                f"a code change hidden behind {hidden_before!r} is accepted"
            )

        line = "fn f<'a>(x: &'a str) -> &'a str { x } // note"
        before, after = line + " " + _CODE + "\n", line + "\n"
        assert comment.shape(_RS, before) == comment.shape(_RS, after), (
            "a comment after a lifetime is a comment"
        )
        assert comment.verdict(_RS, before, after) is None
    finally:
        restore()


# ---------------------------------------------------------------------------
# RU6 -- what cannot be closed is refused, never read as clean
# ---------------------------------------------------------------------------
_UNTERMINATED = (
    "fn f() {} /* a /* b */ c\n",
    'fn f() { let s = "open; }\n',
    'fn f() { let s = r#"open"; }\n',
    "fn f() { let c = '\\n; }\n",
)


def test_ru6_an_unterminated_rust_source_is_refused_and_reported():
    (comment, language), restore = _load(_COMMENT, _LANGUAGE)
    try:
        for tail in _UNTERMINATED:
            before = "// see " + _CODE + "\n" + tail
            after = "// see\n" + tail
            result = comment.verdict(_RS, before, after)
            assert comment.is_refusal(result), (
                f"an unterminated source is refused, never accepted: {tail!r} "
                f"-> {result!r}"
            )
            assert "shape could not be established" in result
            with tempfile.TemporaryDirectory() as tmp:
                _write(tmp, {_RS: "/// " + _FRENCH + "\n" + tail})
                found = language.added_violations(tmp, {_RS: {1}})
                assert [kind for _p, _l, kind, _t in found] == ["unparsable"], (
                    f"an unterminated source is reported, never read: {tail!r}"
                )
        # Control: the same file closed is read, and its French is found.
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, {_RS: "/// " + _FRENCH + "\nfn f() {}\n"})
            found = language.added_violations(tmp, {_RS: {1}})
            assert [kind for _p, _l, kind, _t in found] == ["docstring"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# RU7 -- the language guard's perimeter: one file kind per tree
# ---------------------------------------------------------------------------
def test_ru7_the_language_guard_reads_rust_files_under_rust_and_no_build_output():
    (language,), restore = _load(_LANGUAGE)
    try:
        assert "rust/" in language._SCAN_PATHS
        assert language.unreadable_scan_paths(REPO) == ()
        comment = "// " + _FRENCH + "\nfn f() {}\n"
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, {
                "rust/x/Cargo.toml": "[package]\nname = \"x\"\n",
                "rust/x/src/a.rs": comment,
                "rust/x/src/target/d.rs": comment,
                "rust/x/src/b.py": "# " + _FRENCH + "\nx = 1\n",
                "rust/x/target/debug/c.rs": comment,
                "opti_oignon/e.rs": comment,
                "opti_oignon/f.py": "# " + _FRENCH + "\nx = 1\n",
            })
            counts = language.census_tree(tmp, scan_paths=("opti_oignon/", "rust/"))
            assert counts == {
                "opti_oignon/f.py": 1,
                "rust/x/src/a.rs": 1,
                "rust/x/src/target/d.rs": 1,
            }, (
                "rust/ is read for .rs files and a Python tree for .py files; "
                "the Cargo build directory beside a Cargo.toml is skipped, a "
                "module directory that happens to be called target is not"
            )
            assert language.unreadable_scan_paths(
                tmp, scan_paths=("opti_oignon/", "rust/"),
            ) == ()

            original = language._RUST_READER_PATH
            language._RUST_READER_PATH = Path(tmp) / "absent_reader.py"
            try:
                assert language.unreadable_scan_paths(
                    tmp, scan_paths=("opti_oignon/", "rust/"),
                ) == ("rust/",), (
                    "without its reader the Rust tree is named unreadable, "
                    "never counted clean"
                )
            finally:
                language._RUST_READER_PATH = original
    finally:
        restore()


# ---------------------------------------------------------------------------
# RU8 -- Rust prose is read, one span per passage
# ---------------------------------------------------------------------------
def _prose_source(sentence, words):
    """A Rust source carrying ``sentence`` in each kind of comment once.

    The doc runs carry ``words`` two to a line: no line of a run reads as
    prose alone, so only a run read whole can be charged.
    """
    lines = [words[i:i + 2] for i in range(0, len(words), 2)]
    inner = "".join("//! " + " ".join(pair) + "\n" for pair in lines[:2])
    outer = "".join("/// " + " ".join(pair) + "\n" for pair in lines[2:])
    return (
        inner
        + "\n"
        + outer
        + "pub fn a() {}\n"
        + "\n"
        + "// " + sentence + "\n"
        + "/* outer /* inner */ " + sentence + " */\n"
        + "#[doc = \"" + sentence + "\"]\n"
        + "pub fn b() {}\n"
    )


_FRENCH_WORDS = ["Charge", "la", "valeur", "du",
                 "Retourne", "le", "fichier", "dans", "la", "liste"]
_ENGLISH_WORDS = ["Loads", "the", "value", "of",
                  "Returns", "the", "file", "in", "the", "list"]


def test_ru8_rust_prose_is_read_one_span_per_passage():
    (language,), restore = _load(_LANGUAGE)
    try:
        french = _prose_source(_FRENCH, _FRENCH_WORDS)
        english = _prose_source(_ENGLISH, _ENGLISH_WORDS)
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, {_RS: french, "rust/c/src/en.rs": english})
            counts = language.census_tree(tmp, scan_paths=("rust/",))
        assert counts == {_RS: 5}, (
            "a //! run, a /// run, a // line, a nested block and a #[doc] "
            f"string are five passages: {counts}"
        )
        found = language.find_violations(french, suffix=".rs")
        assert [(line, kind) for line, kind, _text in found] == [
            (1, "docstring"),
            (4, "docstring"),
            (9, "comment"),
            (10, "comment"),
            (11, "docstring"),
        ], found
        assert language.find_violations(english, suffix=".rs") == [], (
            "the same passages in English are not charged"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RU9 -- strings are data
# ---------------------------------------------------------------------------
def test_ru9_rust_strings_are_data_and_never_read_as_prose():
    (language,), restore = _load(_LANGUAGE)
    try:
        prose = "la valeur est dans le fichier de la liste"
        strings = (
            'fn a() -> &\'static str { "' + prose + '" }\n'
            'fn b() -> &\'static str { r#"' + prose + ' " // ' + prose + '"# }\n'
            'fn c() -> &\'static [u8] { b"' + prose + '" }\n'
            'fn d() -> &\'static core::ffi::CStr { c"' + prose + '" }\n'
            "fn e() -> (char, &'static str) { ('\"', \"" + prose + "\") }\n"
            "fn f(x: &'static str) -> String { format!(\"{}{}\", x, "
            "\"il n'est // " + prose + "\") }\n"
        )
        control = "// " + prose + "\nfn g() {}\n"
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, {_RS: strings, "rust/c/src/control.rs": control})
            counts = language.census_tree(tmp, scan_paths=("rust/",))
        assert counts == {"rust/c/src/control.rs": 1}, (
            "the prose in a comment is read, and the same prose inside a "
            f"string, a raw string, a byte string or a C string is data: {counts}"
        )
        # The census leaves an unreadable file out of its count, so the same
        # source is read directly as well: nothing at all is reported,
        # unparsable included.
        assert language.find_violations(strings, suffix=".rs") == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# RU10 -- the added lines of a Rust file are read
# ---------------------------------------------------------------------------
def test_ru10_the_added_lines_of_a_rust_file_are_read():
    (language,), restore = _load(_LANGUAGE)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, {
                _RS: (
                    "//! Engine notes.\n"
                    "/// " + _FRENCH + "\n"
                    "pub fn value() -> u32 {\n"
                    "    7\n"
                    "}\n"
                ),
                "opti_oignon/m.py": "# " + _FRENCH + "\nx = 1\n",
            })
            found = language.added_violations(tmp, {_RS: {2}})
            assert [(path, line, kind) for path, line, kind, _t in found] == [
                (_RS, 2, "docstring"),
            ], found
            assert _FRENCH in found[0][3]
            assert language.added_violations(tmp, {_RS: {4}}) == [], (
                "prose on a line the diff did not add is the census's business"
            )
            found = language.added_violations(tmp, {"opti_oignon/m.py": {1}})
            assert [(path, kind) for path, _l, kind, _t in found] == [
                ("opti_oignon/m.py", "comment"),
            ]
    finally:
        restore()


# ---------------------------------------------------------------------------
# RU11 -- a comment between two tokens keeps them apart
# ---------------------------------------------------------------------------
def test_ru11_a_purged_comment_never_fuses_the_tokens_it_separated():
    (comment,), restore = _load(_COMMENT)
    try:
        before = "pub/* " + _CODE + " */fn f() {}\n"
        fused = "pubfn f() {}\n"
        assert comment.debt_count(before) > comment.debt_count(fused)
        assert comment.verdict(_RS, before, fused) is not None, (
            "removing the comment made one name of two: that is code moving"
        )
        purged = "pub/* */fn f() {}\n"
        assert comment.shape(_RS, before) == comment.shape(_RS, purged), (
            "a purge inside the comment leaves the two tokens as they were"
        )
        assert comment.verdict(_RS, before, purged) is None
    finally:
        restore()


# ---------------------------------------------------------------------------
# RU12 -- a literal's suffix is code, never the prefix of a raw string
# ---------------------------------------------------------------------------
# Each literal below is glued to an ``r``. The compiler reads that ``r`` as
# the literal's suffix, then an ordinary string holding an escaped quote, and
# accepts it all inside a macro's input (rustc 1.92, edition 2021, measured
# on these exact sources). Read as the prefix of a raw string instead, the
# escaped quote closes it, the next string runs on to the following line, and
# ``//x"; 1 }`` becomes a comment that hides the value.
_SUFFIXED = ('"a"r', "'a'r", 'b"a"r', 'r"a"r', 'br#"a"#r')


def _suffixed(glued, value, tail=""):
    """A source whose value follows a literal suffixed with ``r``."""
    return (
        "macro_rules! m { ($($t:tt)*) => {}; }\n"
        "m!(" + glued + '"\\" x");\n'
        + tail
        + 'pub fn f() -> u32 { let _u = "http://x"; ' + value + " }\n"
    )


def test_ru12_a_literal_suffix_is_code_and_never_opens_a_raw_string():
    (comment, language), restore = _load(_COMMENT, _LANGUAGE)
    try:
        for glued in _SUFFIXED:
            one, two = _suffixed(glued, "1"), _suffixed(glued, "2")
            assert comment.comment_free(_RS, one) != comment.comment_free(
                _RS, two,
            ), f"the value after the suffixed {glued!r} is code, not comment"
            before = one + "// note " + _CODE + "\n"
            after = two + "// note\n"
            assert comment.debt_count(before) > comment.debt_count(after)
            # Both sides lex: a refusal would pass the clause below for the
            # wrong reason.
            comment.shape(_RS, before)
            comment.shape(_RS, after)
            assert comment.verdict(_RS, before, after) is not None, (
                f"a code change hidden behind the suffixed {glued!r} is accepted"
            )
            source = _suffixed(glued, "1", tail="// " + _FRENCH + "\n")
            found = language.find_violations(source, suffix=".rs")
            assert [(line, kind) for line, kind, _t in found] == [
                (3, "comment"),
            ], f"a comment after the suffixed {glued!r} is read: {found}"
    finally:
        restore()


# ---------------------------------------------------------------------------
# RU13 -- the source starts where the compiler starts
# ---------------------------------------------------------------------------
_BOM = chr(0xFEFF)


def _body(value):
    """A line whose code holds ``//`` inside a string, and then a value."""
    return 'pub fn f() -> u32 { let _u = "http://x"; ' + value + " }\n"


# Each head compiles (rustc 1.92, measured) and carries a quote the compiler
# never reads as one: on a shebang line that follows a byte-order mark, on a
# shebang line whose next character is not the compiler's whitespace (a file
# separator, which Python's ``isspace`` counts), and inside a comment between
# ``#!`` and the ``[`` of an inner attribute. Read the other way round, that
# quote opens a string that turns every quote after it round.
_HEADS = (
    _BOM + '#!/usr/bin/env run "\n',
    "#!" + chr(0x1C) + '[ "\n',
    '#!/*\n"\n*/[allow(unused)]\n',
)


def test_ru13_the_source_starts_where_the_compiler_starts():
    (comment,), restore = _load(_COMMENT)
    try:
        for head in _HEADS:
            before = head + _body("1") + "// note " + _CODE + "\n"
            after = head + _body("2") + "// note\n"
            assert comment.debt_count(before) > comment.debt_count(after)
            comment.shape(_RS, before)
            comment.shape(_RS, after)
            assert comment.verdict(_RS, before, after) is not None, (
                f"a code change hidden behind the head {head!r} is accepted"
            )

        before = "#! // note " + _CODE + "\n[allow(unused)]\n" + _body("1")
        after = "#! // note\n[allow(unused)]\n" + _body("1")
        assert comment.shape(_RS, before) == comment.shape(_RS, after), (
            "a plain comment between #! and the [ of an attribute is a "
            "comment, not the tail of a shebang line"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RU14 -- only a CR LF pair ends a Rust line
# ---------------------------------------------------------------------------
_CR = chr(13)


def test_ru14_only_a_cr_lf_pair_ends_a_rust_line():
    (comment, language), restore = _load(_COMMENT, _LANGUAGE)
    try:
        crlf = _CR + "\n"
        assert comment.rust_text(("a" + crlf + "b" + _CR + "c").encode()) == (
            "a\nb" + _CR + "c"
        ), "a CR LF pair is a newline, and a lone CR is a character"

        # The compiler keeps the sentence after the CR inside the comment.
        text = (
            "// Plain English note." + _CR + _FRENCH + "\n"
            "pub fn f() -> u32 { 1 }\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / _RS
            path.parent.mkdir(parents=True)
            path.write_bytes(text.encode("utf-8"))
            assert language.census_tree(tmp, scan_paths=("rust/",)) == {_RS: 1}, (
                "the census reads a sentence after a lone CR in its comment"
            )
            found = language.added_violations(tmp, {_RS: {1}})
            assert [(line, kind) for _p, line, kind, _t in found] == [
                (1, "comment"),
            ], f"the added-line check reads it there too: {found}"
            assert comment._worktree_text(path) == text, (
                "the comment-only guard reads the working file as the compiler"
            )

        raw = ("// a" + _CR + '"' + crlf + "fn f() {}" + crlf).encode()

        def run(cmd, **kwargs):
            if kwargs.get("text"):
                # What text mode hands back: universal newlines.
                folded = raw.decode().replace(crlf, "\n").replace(_CR, "\n")
                return types.SimpleNamespace(stdout=folded, returncode=0)
            return types.SimpleNamespace(stdout=raw, returncode=0)

        comment.subprocess = types.SimpleNamespace(run=run)
        assert comment._blob_at("HEAD", _RS) == "// a" + _CR + '"\nfn f() {}\n', (
            "the base side of a Rust file is read as the compiler reads it"
        )
        assert comment._blob_at("HEAD", "frontend/a.ts") == '// a\n"\nfn f() {}\n', (
            "any other file keeps the universal newlines it was read with"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RU15 -- a doc value is prose inside an attribute and data outside one
# ---------------------------------------------------------------------------
def test_ru15_a_doc_value_is_prose_inside_an_attribute_and_data_outside():
    (language,), restore = _load(_LANGUAGE)
    try:
        def read(source):
            return [
                (line, kind)
                for line, kind, _t in language.find_violations(source, suffix=".rs")
            ]

        assert read(
            "pub fn label() -> String {\n"
            '    format!("{doc}", doc = "' + _FRENCH + '")\n'
            "}\n"
        ) == [], "a format! argument named doc is data, like any string"
        assert read(
            '#[doc = concat!("' + _FRENCH + '")]\npub fn f() {}\n'
        ) == [(1, "docstring")], "a concat! doc value is documentation"
        assert read(
            '#![doc = concat!("' + _FRENCH + '")]\npub fn f() {}\n'
        ) == [(1, "docstring")], "an inner doc attribute is read like an outer one"
        assert read(
            '#[cfg_attr(all(), doc = "' + _FRENCH + '")]\npub fn f() {}\n'
        ) == [(1, "docstring")], "a doc value in an attribute's list is read"
        assert read(
            '#[doc = "Returns the value."]\n'
            "pub fn f() -> &'static str { \"" + _FRENCH + '" }\n'
        ) == [], "the doc value ends with its attribute"
    finally:
        restore()


# ---------------------------------------------------------------------------
# RU16 -- a TOML file holding a multi-line string is refused
# ---------------------------------------------------------------------------
def test_ru16_a_toml_file_holding_a_multi_line_string_is_refused():
    (comment,), restore = _load(_COMMENT)
    try:
        toml = "rust/c/Cargo.toml"
        for quote in ('"""', "'''"):
            head = '[package]\nname = "c"\ndescription = ' + quote + "\n"
            before = head + "Core # " + _CODE + " build 1\n" + quote + "\n"
            after = head + "Core # build 2\n" + quote + "\n"
            assert comment.debt_count(before) > comment.debt_count(after)
            result = comment.verdict(toml, before, after)
            assert comment.is_refusal(result), (
                f"a {quote} value whose lines the hash model reads as comments "
                f"is refused, never accepted: {result!r}"
            )
            assert "shape could not be established" in result, result

        before = "# note " + _CODE + '\n[package]\nname = "c"\n'
        after = '# note\n[package]\nname = "c"\n'
        assert comment.verdict(toml, before, after) is None, (
            "a TOML file with no multi-line string keeps its comment model"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RU17 -- every tracked file under rust/ is printable ASCII
# ---------------------------------------------------------------------------
def _unprintable(data):
    """Offsets of the bytes outside printable ASCII, tab and newline."""
    return [
        offset for offset, byte in enumerate(data)
        if not (byte in (9, 10) or 32 <= byte <= 126)
    ]


def test_ru17_every_tracked_file_under_rust_is_printable_ascii():
    listed = _git(REPO, "ls-files", "-z", "--", "rust")
    paths = sorted(item for item in listed.split(chr(0)) if item)
    for expected in ("rust/allium/Cargo.lock", "rust/allium/src/lib.rs",
                     "rust/oo_core/Cargo.toml", "rust/oo_core/src/probes.rs"):
        assert expected in paths, (expected, paths)
    for path in paths:
        assert _unprintable((REPO / path).read_bytes()) == [], (
            f"{path} holds a byte the guards' Rust reading was not measured on"
        )
    for planted in (chr(0xE9).encode(), chr(0xFEFF).encode(), bytes([0]),
                    bytes([13]), bytes([0x1C]), bytes([0x7F])):
        assert _unprintable(b"a" + planted + b"b"), planted


# ---------------------------------------------------------------------------
# RU18-RU21 -- the entry points, end to end over a repository
# ---------------------------------------------------------------------------
def _git(root, *args):
    """Run git in ``root``; its standard output, stripped."""
    return subprocess.run(
        ["git", "-c", "core.autocrlf=false", *args],
        cwd=str(root), capture_output=True, check=True, text=True,
        env={key: value for key, value in os.environ.items()
             if not key.startswith("GIT_")},
    ).stdout.strip()


def _write_bytes(root, files):
    """Write ``{relative path: text}`` under ``root`` as UTF-8, byte for byte."""
    for relative, text in files.items():
        path = Path(root) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))


def _tree_repo(root, base, head=None):
    """A repository at ``root`` whose index holds ``base``, ``head`` written over it.

    Returns the tree ``base`` was written as. The guards take it as their
    base ref, so nothing is ever committed.
    """
    _git(root, "init", "-q")
    _write_bytes(root, base)
    _git(root, "add", "-A")
    tree = _git(root, "write-tree")
    if head:
        _write_bytes(root, head)
    return tree


@contextlib.contextmanager
def _inside(root):
    """Run with ``root`` as the working directory and no git variable set."""
    here = os.getcwd()
    saved = {key: os.environ.pop(key) for key in list(os.environ)
             if key.startswith("GIT_")}
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(here)
        os.environ.update(saved)


def _main(module, argv):
    """``(exit code, standard output)`` of the module's ``main``."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = module.main(argv)
    return code, out.getvalue()


def test_ru18_the_clean_guard_entry_point_charges_an_added_rust_line():
    (clean,), restore = _load(_CLEAN)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tree = _tree_repo(
                tmp,
                {_RS: "pub fn f() {}\n"},
                {_RS: "pub fn f() {}\n// see " + _CODE + "\n"},
            )
            with _inside(tmp):
                code, out = _main(clean, [tree])
        assert code == 1 and _RS + " [session_code]" in out, (
            f"a code added to a Rust file fails the clean guard's run: {out}"
        )
    finally:
        restore()


_PYTHON_TREES = {
    "opti_oignon/a.py": "x = 1\n",
    "tests/a.py": "x = 1\n",
    "scripts/a.py": "x = 1\n",
}
_LANGUAGE_COPY = "_public_language_guard_copy_under_rust_reach"


def _language_guard_in(root):
    """The language guard copied into ``root``, which it then takes as its tree.

    The guard reads the tree it sits in, so a copy sits in a fixture. Its
    Rust reader and the clean guard are copied beside it, as they ship.
    """
    scripts = Path(root) / ".github" / "scripts"
    scripts.mkdir(parents=True)
    for name in ("public_language_guard.py", "comment_only_guard.py",
                 "public_clean_guard.py"):
        shutil.copyfile(_GUARDS / name, scripts / name)
    loaded, restore = isolate(
        targets={_LANGUAGE_COPY: scripts / "public_language_guard.py"},
    )
    return loaded[_LANGUAGE_COPY], restore


def test_ru19_the_language_guard_entry_point_reads_added_rust_lines():
    with tempfile.TemporaryDirectory() as tmp:
        language, restore = _language_guard_in(tmp)
        try:
            tree = _tree_repo(
                tmp,
                {**_PYTHON_TREES, _RS: "pub fn f() {}\n"},
                {_RS: "/// " + _FRENCH + "\npub fn f() {}\n"},
            )
            with _inside(tmp):
                code, out = _main(language, [tree])
        finally:
            restore()
    assert code == 1 and _RS + ":1 [docstring]" in out, (
        f"French added to a Rust file fails the language guard's run: {out}"
    )


def test_ru20_the_language_guard_entry_point_holds_the_rust_census():
    with tempfile.TemporaryDirectory() as tmp:
        language, restore = _language_guard_in(tmp)
        try:
            tree = _tree_repo(
                tmp, {**_PYTHON_TREES, _RS: "// " + _FRENCH + "\npub fn f() {}\n"},
            )
            with _inside(tmp):
                code, out = _main(language, [tree])
        finally:
            restore()
    assert code == 1 and _RS + ": sealed at 0, now carries 1" in out, (
        f"standing French in a Rust file is debt the census charges: {out}"
    )
    assert "on added lines" not in out, "nothing was added: the census alone"

    with tempfile.TemporaryDirectory() as tmp:
        language, restore = _language_guard_in(tmp)
        try:
            tree = _tree_repo(
                tmp, {**_PYTHON_TREES, _RS: "// " + _ENGLISH + "\npub fn f() {}\n"},
            )
            with _inside(tmp):
                code, out = _main(language, [tree])
        finally:
            restore()
    assert code == 0 and ".rs files under rust/" in out, (
        f"a clean run passes and names the Rust tree it read: {out}"
    )


def test_ru21_the_comment_only_entry_point_examines_a_rust_purge():
    (comment,), restore = _load(_COMMENT)
    try:
        # The compiler reads each first line as one comment, the CR inside
        # it: the purge moves nothing, on either side of the diff.
        with tempfile.TemporaryDirectory() as tmp:
            tree = _tree_repo(
                tmp,
                {_RS: "// see " + _CODE + _CR + "note 1\npub fn f() -> u32 { 1 }\n"},
                {_RS: "// see" + _CR + "note 2\npub fn f() -> u32 { 1 }\n"},
            )
            with _inside(tmp):
                code, out = _main(comment, [tree])
        assert code == 0 and "1 of 1 file(s)" in out and "0 not judged" in out, (
            f"the Rust purge is examined and proven: {out}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RU22 -- the diff readers read every added line, and all of it
# ---------------------------------------------------------------------------
_VT, _FF = chr(0x0B), chr(0x0C)


def test_ru22_the_diff_readers_read_every_added_line_whole():
    (clean, language), restore = _load(_CLEAN, _LANGUAGE)
    try:
        # A lone CR, a vertical tab and a form feed are line breaks to text
        # mode or to ``splitlines``, and a line of text opening with ``++ ``
        # is ``+++ `` once the diff adds its own ``+``. The compiler reads
        # all four lines whole, inside the block comment.
        added = [
            "note" + _VT + _CODE + " kept",
            "note" + _FF + _CODE + " kept",
            "note" + _CR + _CODE + " kept",
            "++ " + _CODE + " kept",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tree = _tree_repo(
                tmp,
                {_RS: "pub fn f() {}\n/*\n*/\n"},
                {_RS: "pub fn f() {}\n/*\n" + "\n".join(added) + "\n*/\n"},
            )
            with _inside(tmp):
                pairs = clean._added_lines_with_paths(tree)
        assert pairs == [(_RS, line) for line in added], (
            f"the clean guard reads each added line whole, as its own: {pairs}"
        )

        # A header read inside a hunk hands the rest of the diff to another
        # path: after a form feed that forges a file boundary, to none; after
        # a line opening ``++ b/``, to a file that does not exist.
        with tempfile.TemporaryDirectory() as tmp:
            tree = _tree_repo(
                tmp,
                {_RS: "a\nb\nc\nd\ne\n"},
                {_RS: (
                    "a\n// x" + _FF + "diff --git a/x b/x" + _FF
                    + "+++ /dev/null\nb\nc\n"
                    "++ b/elsewhere.rs\nd\ne\n/// " + _FRENCH + "\n"
                )},
            )
            with _inside(tmp):
                by_path = language._added_lines_by_path(tree)
        assert by_path == {_RS: {2, 5, 8}}, (
            f"the language guard keeps every hunk on its own file: {by_path}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RU23 -- a shebang holding a lone quote is skipped, never refused
# ---------------------------------------------------------------------------
def test_ru23_a_shebang_holding_a_lone_quote_is_skipped_never_refused():
    (comment, language), restore = _load(_COMMENT, _LANGUAGE)
    try:
        # Both compile (rustc 1.92, measured): the first line is a shebang,
        # after a byte-order mark, or with a file separator the compiler
        # does not skip before its ``[``. Read as code, its quote never
        # closes and a sound file is refused.
        for head in (_BOM + '#!/usr/bin/env rust-script "--quiet\n',
                     "#!" + chr(0x1C) + '["\n'):
            source = head + "fn main() {}\n"
            before = source + "// note " + _CODE + "\n"
            assert comment.verdict(_RS, before, source + "// note\n") is None, (
                f"a shebang holding a lone quote is skipped, not refused: {head!r}"
            )
            assert language.find_violations(source, suffix=".rs") == [], (
                f"a shebang holding a lone quote is not unparsable: {head!r}"
            )
    finally:
        restore()


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
