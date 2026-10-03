#!/usr/bin/env python3
"""Comment-only guard: nomenclature may leave a file, nothing else may.

Removing internal nomenclature from the published trees is a large, dull,
mechanical edit spread over hundreds of files -- exactly the shape of diff
nobody reads line by line. The test suites do not cover that risk: they reach
only some of the published trees, and none of them pins a log message, a CLI
banner or an endpoint description. A green run after such an edit means almost
nothing.

So this guard does not test. It PROVES, by construction, and only where the
risk actually is:

    for every file whose nomenclature count went DOWN in this diff,
    the executable shape of the file must be byte-identical before and after.

Files whose count is unchanged are not examined at all, so ordinary work is
never blocked. Files whose count went UP are the public-clean guard's
business, not this one's. The check therefore cannot be inert: it fires on
precisely the operation it exists to make safe, and is silent otherwise.

Two shape provers, because the published trees have nothing in common:

  * Python -- the parsed tree, dumped with docstrings neutralised. Comments
    never reach the parser and docstrings are blanked, so editing either
    leaves the shape untouched; a string literal, a name or a call moves it.

  * Everything else -- the source with exactly the comment byte spans
    removed, found by a per-file state machine that also tracks quotes, so a
    comment marker inside a string or a URL is never mistaken for a comment.
    Rust is read by a lexer of its own, written after the compiler's: its
    block comments nest, a raw string holds any quote behind its hashes, an
    apostrophe opens a char, a lifetime or a label by what follows it, an
    identifier glued to a literal is its suffix, a leading byte-order mark
    and a shebang line are skipped as the compiler skips them, and only a
    CR LF pair ends a line. A comment, string or char it cannot close is
    refused, never guessed. Doc comments are comments there, free like an
    internal docstring; a fenced example in a doc comment of a library crate
    is compiled and run by ``cargo test``, so that freedom is a known gap,
    stated rather than closed. A TOML file holding a multi-line string is
    refused: the hash model cannot tell its lines from comments.

TWO families of docstring are deliberately NOT neutralised, because the
framework publishes both verbatim into the generated API schema: the
docstring of a web route handler, which becomes the endpoint description,
and the docstring of a model class the schema carries, which becomes the
schema description. Both are shipped artefacts rather than internal notes.
Editing one is a real change to a public surface and this guard says so,
loudly, instead of waving it through with the comments.

Which route handlers exist is decidable from one file, so it is decided
here. Which model classes reach the schema is NOT, so it is not guessed: it
is read from the digest recorded by the published-prose guard, which asks
the framework rather than predicting it. That guard remains the outer judge;
this one is the earlier, cheaper signal, and neither is the other's proof.

A file that cannot be parsed or read is REFUSED, never assumed equivalent: a
prover that stays silent on what it failed to understand proves nothing.

The pure helpers (``debt_count``, ``python_shape``, ``comment_free``,
``shape``, ``verdict``, ``rust_tokens``) are import-safe and unit-tested;
``main`` performs the git scan and exits non-zero on any refusal. The
public-language guard loads this file by path for ``rust_tokens`` and
``rust_text``, so the two guards read a Rust file the same way.
Usage: ``comment_only_guard.py [BASE_REF]`` (default base ref: origin/main).
"""

import ast
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

_HERE = Path(__file__).resolve().parent

# Trees this guard covers: the same published trees the clean guard scans.
# The vocabulary is not restated here -- it is imported from the clean guard
# below, so the two can never drift apart. The list of trees is restated, and
# a contract holds it equal to the clean guard's.
_SCAN_PATHS = (
    "opti_oignon/", "tests/", "frontend/", "scripts/", "android/", "rust/",
)

_DEFAULT_BASE_REF = "origin/main"

# Suffix families the non-Python prover understands.
_C_LIKE = frozenset({
    ".ts", ".js", ".mjs", ".cjs", ".kt", ".java", ".css", ".scss",
    ".gradle", ".kts",
})
_MARKUP_LIKE = frozenset({".svelte", ".html"})
_HASH_LIKE = frozenset({".sh", ".bash", ".yml", ".yaml", ".toml", ".cfg",
                        ".ini"})
# Rust is none of the above: see ``rust_tokens``.
_RUST_LIKE = frozenset({".rs"})

# Decorator attributes that mark a function as a web route handler. Its
# docstring is published in the generated API schema, so it is not free.
_ROUTE_DECORATORS = frozenset({
    "get", "post", "put", "delete", "patch", "head", "options", "websocket",
})


class ShapeUnavailable(Exception):
    """The shape of a file could not be computed; it must not be waved on."""


def _load_clean_guard():
    """Import the sibling clean guard by path; its vocabulary is the one."""
    path = _HERE / "public_clean_guard.py"
    spec = importlib.util.spec_from_file_location(
        "_clean_guard_for_comment_only", path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def debt_count(text, clean_guard=None):
    """Number of lines in ``text`` carrying internal nomenclature.

    Delegates to the clean guard's own detector so this guard can never
    disagree with it about what counts.
    """
    guard = clean_guard or _load_clean_guard()
    return len(guard.find_violations(text.splitlines()))


# --------------------------------------------------------------- Python ---

def _route_docstring_ids(tree):
    """Ids of docstring constants belonging to web route handlers."""
    marked = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        is_route = False
        for decorator in node.decorator_list:
            call = decorator.func if isinstance(decorator, ast.Call) \
                else decorator
            if isinstance(call, ast.Attribute) and call.attr in _ROUTE_DECORATORS:
                is_route = True
        if not is_route or not node.body:
            continue
        first = node.body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            marked.add(id(first.value))
    return marked


_DIGEST_NAME = "published_prose.digest"
_UNSET = object()
_DIGEST_CACHE = {}


def published_model_names(path=None):
    """Names of model classes the framework publishes, read from the digest.

    The framework publishes a model class docstring as the description of its
    schema, exactly as it publishes a route handler docstring as the endpoint
    description. Which classes reach the schema is not decidable from one
    file, so it is not guessed here: it is read from the digest recorded
    beside this script by the published-prose guard.

    Returns None when the digest cannot be read. A caller handed None must
    treat EVERY class docstring as published. A prover that cannot establish
    what is published does not get to assume that nothing is.
    """
    if path is None:
        path = Path(__file__).resolve().parent / _DIGEST_NAME
    path = str(path)
    if path in _DIGEST_CACHE:
        return _DIGEST_CACHE[path]
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        names = None
    else:
        names = set()
        for line in text.splitlines():
            parts = line.split(" ")
            if len(parts) == 3 and parts[0] == "schema":
                names.add(parts[1])
    _DIGEST_CACHE[path] = names
    return names


def _model_docstring_ids(tree, published):
    """Ids of docstring constants belonging to published model classes."""
    marked = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or not node.body:
            continue
        if published is not None and node.name not in published:
            continue
        first = node.body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            marked.add(id(first.value))
    return marked


class _BlankDocstrings(ast.NodeTransformer):
    """Replace each internal docstring with a fixed placeholder.

    A published docstring -- on a route handler or on a model class the
    schema carries -- is left exactly as written, so a change to one moves
    the shape and has to be declared rather than absorbed.
    """

    def __init__(self, published):
        self._published = published

    def _blank(self, node):
        self.generic_visit(node)
        body = getattr(node, "body", None)
        if body:
            first = body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                    and id(first.value) not in self._published):
                first.value.value = "<doc>"
        return node

    visit_Module = _blank
    visit_ClassDef = _blank
    visit_FunctionDef = _blank
    visit_AsyncFunctionDef = _blank


def python_shape(text, published_models=_UNSET):
    """Digest of everything in ``text`` that executes.

    Comments never reach the parser; internal docstrings are blanked. What
    remains is the code, its literals and its published descriptions -- of
    endpoints, and of the model schemas the framework builds from classes.

    ``published_models`` names the model classes that reach the schema. Left
    unset it is read from the recorded digest; None means the published set
    is unknown and every class docstring is held published.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError) as exc:
        raise ShapeUnavailable(f"cannot parse: {exc}") from exc
    if published_models is _UNSET:
        published_models = published_model_names()
    return hashlib.md5(
        _shape_dump(tree, published_models).encode()
    ).hexdigest()


def _blanked(tree, published_models):
    """``tree`` with every internal docstring blanked, published ones kept."""
    published = _route_docstring_ids(tree)
    published |= _model_docstring_ids(tree, published_models)
    tree = _BlankDocstrings(published).visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def _shape_dump(tree, published_models):
    """The dump ``python_shape`` digests; shared so provers compare trees."""
    return ast.dump(_blanked(tree, published_models), annotate_fields=True,
                    include_attributes=False)


# ----------------------------------------------------------- everything ---

def _comment_spans(text, markup=False, hash_style=False):
    """Byte spans of comments, as ``{line: [(col0, col1), ...]}``.

    Quote-aware: a comment marker inside a string literal or a URL is not a
    comment. Handles ``//``, ``/* */``, ``#`` and, for markup, ``<!-- -->``.
    """
    spans = {}
    row = col = 0
    row, col = 1, 0
    index, size = 0, len(text)
    state, start, quote = "code", None, ""

    def close(row0, col0, row1, col1):
        for line in range(row0, row1 + 1):
            low = col0 if line == row0 else 0
            high = col1 if line == row1 else 10 ** 9
            spans.setdefault(line, []).append((low, high))

    while index < size:
        char = text[index]
        nxt = text[index + 1] if index + 1 < size else ""
        if state == "code":
            if not hash_style and char == "/" and nxt == "/":
                state, start = "line", (row, col)
                index, col = index + 2, col + 2
                continue
            if not hash_style and char == "/" and nxt == "*":
                state, start = "block", (row, col)
                index, col = index + 2, col + 2
                continue
            if hash_style and char == "#" and (
                col == 0 or text[index - 1] in " \t"
            ):
                # A hash glued to the token before it is part of that token,
                # not a comment opener: the shell parameter count and a
                # fragment inside a bare URL both read as code. Treating one
                # as a comment blinds the prover to every byte that follows
                # on the line, which is exactly where a real edit could hide.
                state, start = "line", (row, col)
                index, col = index + 1, col + 1
                continue
            if markup and text.startswith("<!--", index):
                state, start = "markup", (row, col)
                index, col = index + 4, col + 4
                continue
            if char in "\"'`":
                state, start, quote = "string", (row, col), char
                index, col = index + 1, col + 1
                continue
        elif state == "line":
            if char == "\n":
                close(start[0], start[1], row, col)
                state = "code"
        elif state == "block":
            if char == "*" and nxt == "/":
                close(start[0], start[1], row, col + 2)
                state = "code"
                index, col = index + 2, col + 2
                continue
        elif state == "markup":
            if text.startswith("-->", index):
                close(start[0], start[1], row, col + 3)
                state = "code"
                index, col = index + 3, col + 3
                continue
        elif state == "string":
            if char == "\\":
                index, col = index + 2, col + 2
                continue
            if char == quote:
                state = "code"
            elif char == "\n" and quote != "`":
                state = "code"
        if char == "\n":
            row, col = row + 1, 0
        else:
            col += 1
        index += 1
    if state in ("line", "block", "markup"):
        close(start[0], start[1], row, col)
    return spans


# ----------------------------------------------------------------- Rust ---
#
# Rust is not C, and a C-style model is wrong about it in three places: its
# block comments nest, its raw strings hold any quote behind a count of
# hashes, and its apostrophe opens a char, a lifetime or a loop label
# depending on what follows. Measured on the crates, the C-style model agreed
# with a real Rust lexer on every comment character only by luck -- ninety-
# five lines carry an odd number of lifetime apostrophes and none of them
# happened to carry a comment after them -- and on planted lines it read code
# as comment behind a nested block, behind a raw string holding an odd quote,
# and behind a lifetime followed by a string holding an apostrophe. So Rust
# gets a lexer of its own, written after the compiler's own rules. A byte it
# does not understand stays shape, never comment, and a comment, string or
# char it cannot close raises instead of being guessed.
#
# The compiler's rules include the ones that are easy to miss, because each
# of them, got wrong, turns every quote after it round and so reads code as
# comment: an identifier glued to a literal is that literal's suffix and
# never the prefix of a raw string; the source starts after one byte-order
# mark, and a first line opening with ``#!`` is a shebang unless the next
# token that is not whitespace or a plain comment is ``[``; and only a CR LF
# pair is a line break, a lone CR staying inside the comment that holds it.


class RustUnterminated(ValueError):
    """A Rust comment, string or char runs to the end of the source."""

    def __init__(self, what, offset):
        super().__init__(f"unterminated {what} at offset {offset}")
        self.offset = offset


# Token kinds ``rust_tokens`` reports. Every other byte is code.
RUST_COMMENTS = frozenset({
    "line", "doc_outer", "doc_inner",
    "block", "block_doc_outer", "block_doc_inner",
})
RUST_DOCS = frozenset({
    "doc_outer", "doc_inner", "block_doc_outer", "block_doc_inner",
    "doc_attr",
})

# The characters the compiler counts as whitespace (Unicode's
# Pattern_White_Space). Python's ``str.isspace`` counts more -- the four
# separators from 0x1C to 0x1F among them -- and a character the compiler
# reads as a stray token is not one to skip over.
_RUST_WHITESPACE = frozenset(
    "\t\n\r " + "".join(
        chr(point) for point in (0x0B, 0x0C, 0x85, 0x200E, 0x200F, 0x2028,
                                 0x2029)
    )
)
_RUST_BOM = chr(0xFEFF)


def rust_text(data):
    """A Rust source's bytes as the compiler reads them: ``str``.

    Strict UTF-8, and only a CR LF pair becomes a newline. Python's universal
    newlines would make a lone CR a line break as well, while the compiler
    keeps it inside the comment that carries it: a line comment would end
    early here, and what follows the CR -- a sentence, or a quote that turns
    every string after it round -- would be read as code. Both guards read
    Rust through this function, the base side and the working tree alike.
    """
    return data.decode("utf-8").replace("\r\n", "\n")


def _rust_ident_start(char):
    """True for a character that may begin a Rust identifier."""
    return char == "_" or char.isidentifier()


def _rust_ident_continue(char):
    """True for a character that may continue a Rust identifier."""
    return bool(char) and ("_" + char).isidentifier()


def _rust_line_kind(text, index):
    """Kind of the line comment opened at ``index``: ``////`` is plain."""
    if text.startswith("//!", index):
        return "doc_inner"
    if text.startswith("///", index) and not text.startswith("////", index):
        return "doc_outer"
    return "line"


def _rust_block_kind(text, index):
    """Kind of the block comment opened at ``index``: ``/**/``, ``/***`` plain."""
    if text.startswith("/*!", index):
        return "block_doc_inner"
    if (text.startswith("/**", index)
            and text[index + 3:index + 4] not in ("*", "/")):
        return "block_doc_outer"
    return "block"


def _rust_block_end(text, start):
    """End of the block comment opened at ``start``; nested to any depth."""
    depth, cursor, size = 1, start + 2, len(text)
    while depth:
        if cursor >= size:
            raise RustUnterminated("block comment", start)
        if text.startswith("/*", cursor):
            depth += 1
            cursor += 2
        elif text.startswith("*/", cursor):
            depth -= 1
            cursor += 2
        else:
            cursor += 1
    return cursor


def _rust_start(text):
    """Offset where the compiler starts reading ``text``.

    Past one byte-order mark, which the compiler drops. Then a first line
    opening with ``#!`` is a shebang, skipped to the end of its line, unless
    the first token after the ``#!`` that is neither whitespace nor a plain
    comment is ``[``: then it opens an inner attribute and is read as code.
    Whitespace is the compiler's set, and a doc comment is a token, not
    trivia. A plain block comment left open there ends the compiler's look
    ahead without a ``[``, so the line is a shebang, as it is to the compiler.
    """
    index = 1 if text.startswith(_RUST_BOM) else 0
    if not text.startswith("#!", index):
        return index
    cursor, size = index + 2, len(text)
    while cursor < size:
        if text[cursor] in _RUST_WHITESPACE:
            cursor += 1
        elif (text.startswith("//", cursor)
              and _rust_line_kind(text, cursor) == "line"):
            end = text.find("\n", cursor)
            cursor = size if end == -1 else end
        elif (text.startswith("/*", cursor)
              and _rust_block_kind(text, cursor) == "block"):
            try:
                cursor = _rust_block_end(text, cursor)
            except RustUnterminated:
                break
        else:
            break
    if text.startswith("[", cursor):
        return index
    end = text.find("\n", index)
    return size if end == -1 else end


def _rust_suffix(text, cursor):
    """End of the suffix glued to a literal that ends at ``cursor``.

    The compiler reads an identifier right after a literal as the literal's
    suffix -- ``1u8``, and on a string ``"a"r``, which it rejects later, if
    at all: inside a macro's input it never does. Read as a word instead,
    an ``r``, ``br`` or ``cr`` there would open a raw string the compiler
    never sees, and every quote after it would change sides.
    """
    size = len(text)
    if cursor < size and _rust_ident_start(text[cursor]):
        cursor += 1
        while cursor < size and _rust_ident_continue(text[cursor]):
            cursor += 1
    return cursor


def _rust_string(text, start, cursor, tokens, kind):
    """Lex a string whose body begins at ``cursor``; return where it ends."""
    size = len(text)
    while cursor < size:
        char = text[cursor]
        if char == '"':
            tokens.append((kind, start, cursor + 1))
            return _rust_suffix(text, cursor + 1)
        cursor += 2 if char == "\\" else 1
    raise RustUnterminated("string", start)


def _rust_raw(text, start, cursor, tokens, kind):
    """Lex a raw string whose hashes begin at ``cursor``; None if it is not one."""
    hashes = cursor
    while text.startswith("#", hashes):
        hashes += 1
    if not text.startswith('"', hashes):
        return None
    close = '"' + "#" * (hashes - cursor)
    stop = text.find(close, hashes + 1)
    if stop == -1:
        raise RustUnterminated("raw string", start)
    end = stop + len(close)
    tokens.append((kind, start, end))
    return _rust_suffix(text, end)


def _rust_single_quoted(text, quote):
    """End of the char or byte char whose opening quote is at ``quote``."""
    size = len(text)
    cursor = quote + 1
    first = text[cursor:cursor + 1]
    if first and first != "\\" and text.startswith("'", cursor + 1):
        return cursor + 2
    while cursor < size:
        char = text[cursor]
        if char == "'":
            return cursor + 1
        if char == "/" or (char == "\n" and not text.startswith("'", cursor + 1)):
            break
        cursor += 2 if char == "\\" else 1
    raise RustUnterminated("char", quote)


def _rust_quote(text, start, tokens):
    """Lex what an apostrophe at ``start`` opens; return where it ends.

    A char when an escape follows, or one character and a closing quote,
    and then its suffix; a lifetime or a label when a name follows and no
    quote closes it; a raw lifetime, ``'r#name``, whole and never closed; a
    char spelled with a whole name when a quote does close it (the compiler
    reads it so, with no suffix, then rejects it). Only a char is a token: a
    lifetime is code.
    """
    size = len(text)
    first = text[start + 1:start + 2]
    second = text[start + 2:start + 3]
    if second != "'" and first and (_rust_ident_start(first) or "0" <= first <= "9"):
        raw = (first == "r" and second == "#"
               and _rust_ident_start(text[start + 3:start + 4]))
        cursor = start + 4 if raw else start + 2
        while cursor < size and _rust_ident_continue(text[cursor]):
            cursor += 1
        if not raw and text.startswith("'", cursor):
            tokens.append(("char", start, cursor + 1))
            return cursor + 1
        return cursor
    end = _rust_single_quoted(text, start)
    tokens.append(("char", start, end))
    return _rust_suffix(text, end)


def rust_tokens(text):
    """``[(kind, start, end), ...]`` for every comment and literal in Rust.

    Offsets index ``text``. Comment kinds: ``line`` (``//``, and ``////``
    and more), ``doc_outer`` (``///``), ``doc_inner`` (``//!``), ``block``,
    ``block_doc_outer`` (``/**``, but not ``/**/`` nor ``/***``) and
    ``block_doc_inner`` (``/*!``); block comments nest to any depth. Literal
    kinds: ``string`` (with the ``b`` and ``c`` prefixes), ``raw_string``
    (``r``, ``br``, ``cr``, any count of hashes), ``char`` (and byte char),
    and ``doc_attr`` for a string, raw or not, inside the value of a ``doc``
    attribute -- ``#[doc = ...]``, ``#![doc = ...]``, or ``doc = ...`` in
    the list of an attribute such as ``cfg_attr``, up to the comma or the
    bracket that ends it, so a ``concat!`` of strings is doc as a whole; a
    ``doc = "..."`` anywhere else, a named ``format!`` argument, is data. A
    literal's suffix, a lifetime, a label and a raw identifier are code, and
    so are a leading byte-order mark and a shebang line.

    Raises ``RustUnterminated`` for a block comment, string, raw string or
    char that is never closed.
    """
    tokens, recent = [], []
    # One flag per open bracket, parenthesis or brace: whether it lies
    # inside an attribute. And the depth of the doc value being read.
    inside, doc = [], None
    index, size = _rust_start(text), len(text)
    while index < size:
        char = text[index]
        if text.startswith("//", index):
            end = text.find("\n", index)
            end = size if end == -1 else end
            tokens.append((_rust_line_kind(text, index), index, end))
            index = end
            continue
        if text.startswith("/*", index):
            end = _rust_block_end(text, index)
            tokens.append((_rust_block_kind(text, index), index, end))
            index = end
            continue
        documented = doc is not None
        if char == '"':
            index = _rust_string(text, index, index + 1, tokens,
                                 "doc_attr" if documented else "string")
            recent.append('"')
            continue
        if char == "'":
            index = _rust_quote(text, index, tokens)
            recent.append("'")
            continue
        if "0" <= char <= "9" or _rust_ident_start(char):
            end = index + 1
            while end < size and _rust_ident_continue(text[end]):
                end += 1
            word = text[index:end]
            if word in ("r", "br", "cr") and text[end:end + 1] in ('"', "#"):
                raw = _rust_raw(text, index, end, tokens,
                                "doc_attr" if documented else "raw_string")
                if raw is not None:
                    recent.append('"')
                    index = raw
                    continue
                if (word == "r" and text.startswith("#", end)
                        and _rust_ident_start(text[end + 1:end + 2])):
                    # A raw identifier, ``r#name``: code.
                    end += 1
                    while end < size and _rust_ident_continue(text[end]):
                        end += 1
                    word = text[index + 2:end]
            elif word in ("b", "c") and text.startswith('"', end):
                index = _rust_string(text, index, end + 1, tokens,
                                     "doc_attr" if documented else "string")
                recent.append('"')
                continue
            elif word == "b" and text.startswith("'", end):
                stop = _rust_single_quoted(text, end)
                tokens.append(("char", index, stop))
                recent.append("'")
                index = _rust_suffix(text, stop)
                continue
            recent.append(word)
            index = end
            continue
        if char in "[({":
            opens = char == "[" and (recent[-1:] == ["#"]
                                     or recent[-2:] == ["#", "!"])
            inside.append(opens or (bool(inside) and inside[-1]))
        elif char in "])}":
            if inside:
                inside.pop()
            if doc is not None and len(inside) < doc:
                doc = None
        elif char == ",":
            if doc is not None and len(inside) == doc:
                doc = None
        elif (char == "=" and not text.startswith("=", index + 1)
              and inside and inside[-1] and recent[-1:] == ["doc"]
              and recent[-2:-1] in (["["], ["("], [","])):
            doc = len(inside)
        if not char.isspace():
            recent.append(char)
        if len(recent) > 8:
            del recent[:-4]
        index += 1
    return tokens


def _rust_comment_free_lines(text):
    """Rust ``text`` as lines with every comment removed, trailing space cut.

    Lines are split on newlines alone, as the compiler counts them. A comment
    that held a newline leaves its newlines, so no line moves. A comment
    between two tokens it separated leaves a space, so the purge cannot fuse
    two names into one and call that no change.
    """
    try:
        tokens = rust_tokens(text)
    except RustUnterminated as exc:
        raise ShapeUnavailable(f"cannot lex: {exc}") from exc
    pieces, last = [], 0
    for kind, start, end in tokens:
        if kind not in RUST_COMMENTS:
            continue
        pieces.append(text[last:start])
        breaks = text.count("\n", start, end)
        if breaks:
            pieces.append("\n" * breaks)
        elif (start and not text[start - 1].isspace()
              and end < len(text) and not text[end].isspace()):
            pieces.append(" ")
        last = end
    pieces.append(text[last:])
    return [line.rstrip() for line in "".join(pieces).split("\n")]


def _comment_free_lines(path, text):
    """``text`` as lines with every comment byte removed, trailing space cut.

    An unrecognised suffix gets no comment model at all, so every byte stays
    shape. That is the fail-closed direction: a file this guard cannot even
    tokenise is never treated as though its comments were understood.

    A TOML file holding a multi-line string is refused instead: the hash
    model ends every string at its line, so a ``#`` on a later line of a
    triple-quoted value would be read as a comment while TOML reads it as
    the value itself.
    """
    suffix = Path(path).suffix
    if suffix in _RUST_LIKE:
        return _rust_comment_free_lines(text)
    if suffix in _MARKUP_LIKE:
        spans = _comment_spans(text, markup=True)
    elif suffix in _C_LIKE:
        spans = _comment_spans(text)
    elif suffix in _HASH_LIKE:
        if suffix == ".toml" and ('"""' in text or "'''" in text):
            raise ShapeUnavailable("a multi-line TOML string is not modelled")
        spans = _comment_spans(text, hash_style=True)
    else:
        spans = {}
    kept = []
    for row, line in enumerate(text.splitlines(), 1):
        cuts = spans.get(row)
        if cuts:
            line = "".join(
                char for col, char in enumerate(line)
                if not any(low <= col < high for low, high in cuts)
            )
        kept.append(line.rstrip())
    return kept


def comment_free(path, text):
    """Digest of ``text`` with every comment byte removed."""
    return hashlib.md5(
        "\n".join(_comment_free_lines(path, text)).encode()
    ).hexdigest()


def shape(path, text):
    """Digest of what must not move when nomenclature leaves ``path``."""
    if Path(path).suffix == ".py":
        return python_shape(text)
    return comment_free(path, text)


def _masked_python_shape(text, published_models=_UNSET):
    """AST dump with every live string masked, plus those strings in order.

    Internal docstrings are blanked first, exactly as ``python_shape`` does,
    so they stay free. Every remaining string constant -- runtime literal or
    published docstring -- is collected and replaced by one placeholder, so
    the dump pins everything else: identifiers, calls, structure, numbers.
    """
    if published_models is _UNSET:
        published_models = published_model_names()
    tree = ast.parse(text)
    published = (_route_docstring_ids(tree)
                 | _model_docstring_ids(tree, published_models))
    tree = _BlankDocstrings(published).visit(tree)
    ast.fix_missing_locations(tree)
    strings = []

    class _Mask(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value != "<doc>":
                strings.append(node.value)
                node.value = "<s>"
            return node

    tree = _Mask().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, annotate_fields=True, include_attributes=False), strings


def string_purge_equivalent(before, after, published_models=_UNSET,
                            clean_guard=None):
    """``None`` when the only delta is nomenclature leaving string constants.

    Three refusals, each on its own ground: something other than a string
    constant moved; a changed string still carries nomenclature; a changed
    string carried none to begin with. Equal masked dumps guarantee that the
    two string lists pair position for position, so the walk is total.

    The prover does not judge whether a message still says what it should;
    that stays with human review. What it proves is narrower and mechanical:
    the delta cannot hide a change outside the purged strings, and every
    purged string went from carrying nomenclature to carrying none.
    """
    guard = clean_guard or _load_clean_guard()
    dump_before, old_strings = _masked_python_shape(before, published_models)
    dump_after, new_strings = _masked_python_shape(after, published_models)
    if dump_before != dump_after:
        return "something other than a string constant moved"
    for old, new in zip(old_strings, new_strings):
        if old == new:
            continue
        if not guard.find_violations(old.split("\n")):
            return "a changed string carried no nomenclature"
        if guard.find_violations(new.split("\n")):
            return "a changed string still carries nomenclature"
    return None


# ------------------------------------------------------------ renaming ---
#
# Stripping nomenclature out of identifiers moves the shape by construction,
# so the two provers above refuse the whole of that edit. This one accepts it
# on a narrow proof: the difference is a substitution of identifiers that
# carry nomenclature by identifiers that do not, and nothing else.
#
# The proof is by reconstruction, never by inspection. A candidate map is
# read off the two trees, checked to be a function and injective, and then
# APPLIED to the before side; what it produces must equal the after side
# exactly. A map read wrongly therefore cannot buy an acceptance -- it simply
# fails to reproduce the file. Every identifier slot this module does not
# know about is left out of the map, which can only cause a refusal.

# Node fields holding a plain identifier. Anything absent here is not
# substitutable, so a change in it lands in the residue and is refused.
_ID_FIELDS = {
    ast.Name: ("id",),
    ast.Attribute: ("attr",),
    ast.arg: ("arg",),
    ast.FunctionDef: ("name",),
    ast.AsyncFunctionDef: ("name",),
    ast.ClassDef: ("name",),
    ast.alias: ("name", "asname"),
    ast.keyword: ("arg",),
    ast.ExceptHandler: ("name",),
}

# ``global x, y`` and ``nonlocal x`` hold plain identifiers in a LIST rather
# than in a field of their own. They bind the very names a Name node reads, so
# leaving them out does not merely miss a rename: the two trees stop pairing
# and a pure rename is refused.
_ID_LISTS = {
    ast.Global: "names",
    ast.Nonlocal: "names",
}


_ATTRIBUTE = "attribute"
_VARIABLE = "variable"

_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _slot_space(node, field, in_class_body):
    """The namespace an identifier slot binds or reads in.

    Decided by where Python actually binds the name, never by the syntax that
    spells it. Two names may converge on one only when they never denoted the
    same binding, so the question is always: same namespace or not.

      * ``self.x`` and a method ``def x`` are the SAME namespace. Both are
        looked up on the object, so renaming both onto one name is a merge.
      * A ``def`` or ``class`` at module level, or nested inside a function,
        binds in the ordinary variable space -- exactly as an assignment
        does. Definition names are therefore NOT a namespace of their own.
        Holding them apart would accept a rename that merges a definition
        with a local by shadowing, and the reconstruction cannot catch that:
        the after file really does contain both names, so it rebuilds
        perfectly while two distinct bindings have silently become one.
      * A parameter and a local of the same name are one binding, so
        arguments belong with plain names. ``keyword.arg`` names a parameter
        of the callee rather than binding anything here; it is left in the
        variable space, which is stricter than it needs to be and therefore
        safe.
    """
    if isinstance(node, ast.Attribute) and field == "attr":
        return _ATTRIBUTE
    if isinstance(node, _DEFINITIONS) and field == "name":
        return _ATTRIBUTE if in_class_body else _VARIABLE
    return _VARIABLE


def _identifier_slots(tree):
    """Every ``(node, field, space)`` holding an identifier, depth first."""
    slots = []

    def walk(node, in_class_body):
        for field in _ID_FIELDS.get(type(node), ()):
            if isinstance(getattr(node, field, None), str):
                slots.append((node, field,
                              _slot_space(node, field, in_class_body)))
        field = _ID_LISTS.get(type(node))
        if field:
            for index in range(len(getattr(node, field, ()))):
                slots.append((node, (field, index), _VARIABLE))
        inside = isinstance(node, ast.ClassDef)
        for child in ast.iter_child_nodes(node):
            walk(child, inside)

    walk(tree, False)
    return slots


def _alias_nodes(tree):
    """Import nodes, depth first, so two trees pair theirs positionally."""
    found = []

    def walk(node):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.append(node)
        for child in ast.iter_child_nodes(node):
            walk(child)

    walk(tree)
    return found


def _slot_get(node, field):
    """Read an identifier slot, field or list position alike."""
    if isinstance(field, tuple):
        return getattr(node, field[0])[field[1]]
    return getattr(node, field)


def _slot_set(node, field, value):
    """Write an identifier slot, field or list position alike."""
    if isinstance(field, tuple):
        getattr(node, field[0])[field[1]] = value
    else:
        setattr(node, field, value)


def _pairing_dump(text):
    """Dump with identifiers and strings masked, plus the identifiers.

    Import alias lists are sorted before masking, so a rename that changes
    where a name sorts still pairs its identifiers correctly. The sorting is
    only how the CANDIDATE map is read; the map is verified against the
    unsorted trees afterwards, where a re-sort is refused like any other
    movement.

    Every string is masked, docstrings included, so this pairing does not
    depend on the published set -- which is itself resolved through the map
    once the map is known.
    """
    tree = ast.parse(text)
    for node in _alias_nodes(tree):
        node.names.sort(key=lambda a: (a.name, a.asname or ""))
    names = []
    spaces = []
    for node, field, space in _identifier_slots(tree):
        names.append(_slot_get(node, field))
        spaces.append(space)
        _slot_set(node, field, "<id>")

    class _Mask(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = "<s>"
            return node

    tree = _Mask().visit(tree)
    ast.fix_missing_locations(tree)
    dump = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return dump, names, spaces, tree


def _substitute(text, mapping):
    """Apply the map inside a literal, longest source first."""
    for old in sorted(mapping, key=len, reverse=True):
        text = text.replace(old, mapping[old])
    return text


def _renamed_tree(text, maps, literal_map):
    """``text`` parsed with each namespace's map applied, and literals too.

    A literal belongs to no namespace, so it is substituted through the union
    of the maps -- which the caller has already proved unambiguous.
    """
    tree = ast.parse(text)
    for node, field, space in _identifier_slots(tree):
        value = _slot_get(node, field)
        if value in maps[space]:
            _slot_set(node, field, maps[space][value])

    class _Sub(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = _substitute(node.value, literal_map)
            return node

    tree = _Sub().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def _resort_reason(mapped, after):
    """Name an alias re-sort, when that is what the two trees disagree on."""
    for one, two in zip(_alias_nodes(mapped), _alias_nodes(after)):
        first = [(a.name, a.asname) for a in one.names]
        second = [(a.name, a.asname) for a in two.names]
        if first != second and sorted(first) == sorted(second):
            return "an import alias sequence was re-sorted"
    return None


def _differences(one, two, out, limit=6):
    """Short descriptions of where two blanked trees disagree."""
    if len(out) >= limit:
        return
    if type(one) is not type(two):
        out.append(f"{type(one).__name__} became {type(two).__name__}")
        return
    for field, left in ast.iter_fields(one):
        right = getattr(two, field, None)
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                longer = right if len(right) > len(left) else left
                extra = [type(n).__name__ for n in longer[min(len(left),
                                                              len(right)):]
                         if isinstance(n, ast.AST)]
                out.append(f"{type(one).__name__}.{field} changed length"
                           + (f" ({', '.join(extra)})" if extra else ""))
            for x, y in zip(left, right):
                if isinstance(x, ast.AST) and isinstance(y, ast.AST):
                    _differences(x, y, out, limit)
                elif x != y:
                    out.append(f"{type(one).__name__}.{field}: {x!r} -> {y!r}")
        elif isinstance(left, ast.AST) and isinstance(right, ast.AST):
            _differences(left, right, out, limit)
        elif left != right:
            out.append(f"{type(one).__name__}.{field}: {left!r} -> {right!r}")
        if len(out) >= limit:
            return


_NO_RENAME = "no identifier was renamed"


def rename_equivalent(before, after, published_models=_UNSET,
                      clean_guard=None):
    """``None`` when the whole delta is a proven identifier substitution.

    The map must be a function and injective, every source must carry
    nomenclature and no target may. The map is then applied to the before
    side -- to identifiers and, exactly as the string purge treats them, to
    string literals -- and the result must have the same shape as the after
    side. Anything the substitution does not reproduce is listed and refused.

    Returns the sentinel ``_NO_RENAME`` when no identifier moved at all, so a
    caller can report the more accurate refusal of whichever prover owns the
    delta instead of this one's.
    """
    guard = clean_guard or _load_clean_guard()
    if published_models is _UNSET:
        published_models = published_model_names()

    dump_before, old_names, spaces, masked_before = _pairing_dump(before)
    dump_after, new_names, _, masked_after = _pairing_dump(after)
    if dump_before != dump_after:
        # Identifiers and strings are masked on both sides here, so what is
        # left to differ is exactly what no substitution could explain. That
        # list is the specification of the extraction the file still owes.
        found = []
        _differences(masked_before, masked_after, found)
        listed = "; ".join(found) if found else "the structure moved"
        return f"differences not attributable to a substitution: {listed}"

    maps = {_ATTRIBUTE: {}, _VARIABLE: {}}
    unchanged = {_ATTRIBUTE: set(), _VARIABLE: set()}
    for space, old, new in zip(spaces, old_names, new_names):
        if old == new:
            unchanged[space].add(old)
            continue
        if maps[space].setdefault(old, new) != new:
            return (f"the identifier map is not a function ({old} goes to "
                    f"both {maps[space][old]} and {new})")
    if not any(maps.values()):
        return _NO_RENAME

    # Injectivity is asked of each namespace separately: a collision only
    # merges two things when the two names denoted one binding to begin with.
    for space, mapping in sorted(maps.items()):
        targets = list(mapping.values())
        if len(set(targets)) != len(targets) or unchanged[space] & set(targets):
            return (f"the identifier map is not injective in the {space} "
                    "namespace (two names collapse onto one)")

    # Literals have no namespace, so the union has to be unambiguous.
    literal_map = {}
    for mapping in maps.values():
        for old, new in mapping.items():
            if literal_map.setdefault(old, new) != new:
                return (f"{old} is renamed differently in different "
                        "namespaces, so a literal carrying it is ambiguous")

    for old, new in sorted(literal_map.items()):
        if not guard.find_violations([old]):
            return f"a renamed identifier carried no nomenclature ({old} -> {new})"
        if guard.find_violations([new]):
            return f"a renamed identifier still carries nomenclature ({new})"

    mapped = _blanked(_renamed_tree(before, maps, literal_map),
                      published_models)
    target = _blanked(ast.parse(after), published_models)
    if (ast.dump(mapped, annotate_fields=True, include_attributes=False)
            == ast.dump(target, annotate_fields=True,
                        include_attributes=False)):
        return None

    resort = _resort_reason(mapped, target)
    if resort:
        return resort
    found = []
    _differences(mapped, target, found)
    if found and all(d.startswith("Constant.value:") for d in found):
        return ("a changed string is not explained by the identifier map "
                f"({found[0].split(': ', 1)[1]})")
    listed = "; ".join(found) if found else "the shapes differ"
    return f"differences not attributable to a substitution: {listed}"


class _CannotJudge:
    """The verdict for a file this guard has no means to attribute.

    Deliberately NOT ``None`` and deliberately truthy. A falsy value would
    slip through ``if reason:`` in the caller and become a silent
    acceptance -- an absence of checking recorded as a check that passed,
    which is the one outcome this whole guard exists to prevent.

    Deliberately not a string either. The weak spelling ``reason is not
    None`` is satisfied by a non-judgement, so a contract written that way
    would go on passing while proving nothing. Callers and contracts must
    ask ``is_refusal`` instead, which this value answers False to.
    """

    __slots__ = ()

    def __bool__(self):
        return True

    def __repr__(self):
        return "CANNOT_JUDGE"


CANNOT_JUDGE = _CannotJudge()


def is_refusal(result):
    """True only for an actual refusal, never for a non-judgement."""
    return result is not None and result is not CANNOT_JUDGE


def line_purge_equivalent(path, before, after, clean_guard=None):
    """``None`` when a non-Python delta is a nomenclature purge, line by line.

    Outside Python there is no parse tree, so the finest grain available is
    the comment-stripped line. The test is the string purge's, coarsened to
    that grain: every line that changed must have carried nomenclature and
    must carry none now. It proves less than its Python counterpart -- it
    cannot pin that only a literal moved within the line -- so it is only
    ever the difference between an acceptance and a NON-judgement here,
    never between an acceptance and a refusal.
    """
    guard = clean_guard or _load_clean_guard()
    old = _comment_free_lines(path, before)
    new = _comment_free_lines(path, after)
    if len(old) != len(new):
        return "the number of lines outside comments changed"
    for one, two in zip(old, new):
        if one == two:
            continue
        if not guard.find_violations([one]):
            return "a changed line carried no nomenclature"
        if guard.find_violations([two]):
            return "a changed line still carries nomenclature"
    return None


def proven_rename_map(before, after, published_models=_UNSET,
                      clean_guard=None):
    """The substitution behind an accepted rename, or ``None``.

    Separate from ``rename_equivalent`` on purpose: that function's return
    value already carries three meanings, and a fourth would make it
    unreadable. This one answers a single question with a single type.
    """
    if rename_equivalent(before, after, published_models, clean_guard) is not None:
        return None
    maps = {_ATTRIBUTE: {}, _VARIABLE: {}}
    dump_before, old_names, spaces, _ = _pairing_dump(before)
    _, new_names, _, _ = _pairing_dump(after)
    for space, old, new in zip(spaces, old_names, new_names):
        if old != new:
            maps[space][old] = new
    merged = {}
    for mapping in maps.values():
        merged.update(mapping)
    return merged


def format_rename_map(mapping):
    """The acceptance line's map: ``k identifier(s): old->new, ...``.

    Empty renders empty. An acceptance that renamed nothing must not read as
    though it had proved a rename.
    """
    if not mapping:
        return ""
    pairs = ", ".join(f"{old}->{new}" for old, new in sorted(mapping.items()))
    return f"{len(mapping)} identifier(s): {pairs}"


def verdict(path, before, after, clean_guard=None):
    """Return ``None`` if acceptable, ``CANNOT_JUDGE`` if unattributable,
    else a one-line reason to refuse.

    A file is examined only when its nomenclature count fell. Anything else
    -- unchanged, or risen -- is not this guard's business.
    """
    guard = clean_guard or _load_clean_guard()
    if debt_count(after, guard) >= debt_count(before, guard):
        return None
    try:
        if shape(path, before) == shape(path, after):
            return None
    except ShapeUnavailable as exc:
        return f"shape could not be established ({exc})"
    if Path(path).suffix == ".py":
        reason = string_purge_equivalent(before, after, clean_guard=guard)
        if reason is None:
            return None
        renamed = rename_equivalent(before, after, clean_guard=guard)
        if renamed is None:
            return None
        # Whichever prover owns the delta gives the more useful refusal: the
        # purge's when no identifier moved, the rename's when one did.
        if renamed != _NO_RENAME:
            reason = renamed
        return ("nomenclature was removed AND the executable shape moved "
                f"({reason})")
    # Outside Python the shape moved and there is no analyser to attribute
    # the movement. A proven line-level purge is still an acceptance; what
    # is left is not a refusal, because the guard has established that
    # something changed, not that the change is wrong.
    if line_purge_equivalent(path, before, after, guard) is None:
        return None
    return CANNOT_JUDGE


# ------------------------------------------------------------------ main ---

class BaseUnreadable(Exception):
    """git could not read a file at the base it was asked for."""


def _changed_paths(base_ref):
    """Paths changed in the diff over the scan trees, post-image names.

    ``None`` when git cannot produce the diff: an empty diff and a failed
    one are not the same answer, and only the first is a pass.
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=d", "--no-color",
         base_ref, "--", *_SCAN_PATHS],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return None
    return [line for line in result.stdout.splitlines() if line.strip()]


def _blob_at(base_ref, path):
    """File content at ``base_ref``; ``None`` when it did not exist there.

    A Rust file is read as the compiler reads it (``rust_text``); every
    other file keeps the universal newlines it has always been read with.

    Whether the path existed is asked of the base's tree first, so that a
    read git fails is never taken for a file the base never had: that
    raises ``BaseUnreadable``.
    """
    rust = Path(path).suffix in _RUST_LIKE
    listed = subprocess.run(
        ["git", "ls-tree", "--full-tree", "--name-only", "-z", base_ref,
         "--", path],
        capture_output=True, check=False,
    )
    if listed.returncode != 0:
        raise BaseUnreadable(f"git could not list it at {base_ref}")
    if not listed.stdout:
        return None
    result = subprocess.run(
        ["git", "show", f"{base_ref}:{path}"],
        capture_output=True, text=not rust, check=False,
    )
    if result.returncode != 0:
        raise BaseUnreadable(f"git could not read it at {base_ref}")
    return rust_text(result.stdout) if rust else result.stdout


def _worktree_text(path):
    """The working-tree file at ``path``, read as ``_blob_at`` reads the base."""
    if Path(path).suffix in _RUST_LIKE:
        return rust_text(Path(path).read_bytes())
    return Path(path).read_text(encoding="utf-8")


def main(argv=None):
    """Scan the diff and exit non-zero on any refusal."""
    argv = list(sys.argv[1:] if argv is None else argv)
    base_ref = argv[0] if argv else _DEFAULT_BASE_REF
    clean_guard = _load_clean_guard()

    paths = _changed_paths(base_ref)
    if paths is None:
        print(f"comment-only guard: FAILED -- git could not diff against base {base_ref}")
        return 1

    refusals = []
    unjudged = []
    renamed = []
    unread = []
    examined = compared = new = 0
    for path in paths:
        try:
            before = _blob_at(base_ref, path)
        except BaseUnreadable as exc:
            unread.append((path, str(exc)))
            continue
        if before is None:
            new += 1
            continue
        try:
            after = _worktree_text(path)
        except (OSError, UnicodeDecodeError):
            continue
        compared += 1
        if debt_count(after, clean_guard) >= debt_count(before, clean_guard):
            continue
        examined += 1
        reason = verdict(path, before, after, clean_guard)
        if reason is CANNOT_JUDGE:
            unjudged.append(path)
        elif is_refusal(reason):
            refusals.append((path, reason))
        elif path.endswith(".py"):
            mapping = proven_rename_map(before, after, clean_guard=clean_guard)
            if mapping:
                renamed.append((path, format_rename_map(mapping)))

    if unread:
        print("comment-only guard: FAILED -- git could not read these files "
              "at the base:")
        for path, why in unread:
            print(f"  {path}: {why}")
        return 1

    # Printed before any verdict, so an absence of checking is never folded
    # into a line that reads as a check that passed.
    if unjudged:
        print("comment-only guard: NOT JUDGED -- the shape moved and no "
              "prover could attribute it, the check belongs elsewhere:")
        for path in unjudged:
            print(f"  {path}")

    # An acceptance that does not say what it accepted asks to be trusted
    # rather than read. Every proven rename names its substitution.
    if renamed:
        print("comment-only guard: PROVEN RENAME -- accepted, and here is "
              "what moved:")
        for path, line in renamed:
            print(f"  {path}: {line}")

    if not refusals:
        print(
            "comment-only guard: "
            f"{compared} changed file(s) compared, {new} new; "
            f"{examined - len(unjudged)} of {examined} file(s) shed "
            "nomenclature within an unchanged executable shape, a proven "
            "string purge, a proven rename or a proven line purge "
            f"(base {base_ref}); "
            f"{len(unjudged)} not judged; "
            "published prose still answers to its own digest"
        )
        return 0

    print("comment-only guard: FAILED -- nomenclature left, but so did more:")
    for path, reason in refusals:
        print(f"  {path}: {reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
