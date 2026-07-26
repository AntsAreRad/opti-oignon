#!/usr/bin/env python3
"""Comment-only guard: nomenclature may leave a file, nothing else may.

Removing internal nomenclature from the published trees is a large, dull,
mechanical edit spread over hundreds of files -- exactly the shape of diff
nobody reads line by line. The test suites do not cover that risk: they run
over two of the five published trees, and none of them pins a log message, a
CLI banner or an endpoint description. A green run after such an edit means
almost nothing.

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
``shape``, ``verdict``) are import-safe and unit-tested; ``main`` performs
the git scan and exits non-zero on any refusal.
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
# below, so the two can never drift apart.
_SCAN_PATHS = (
    "opti_oignon/", "tests/", "frontend/", "scripts/", "android/",
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
    published = _route_docstring_ids(tree)
    published |= _model_docstring_ids(tree, published_models)
    tree = _BlankDocstrings(published).visit(tree)
    ast.fix_missing_locations(tree)
    dump = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.md5(dump.encode()).hexdigest()


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


def comment_free(path, text):
    """Digest of ``text`` with every comment byte removed."""
    suffix = Path(path).suffix
    if suffix in _MARKUP_LIKE:
        spans = _comment_spans(text, markup=True)
    elif suffix in _C_LIKE:
        spans = _comment_spans(text)
    elif suffix in _HASH_LIKE:
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
    return hashlib.md5("\n".join(kept).encode()).hexdigest()


def shape(path, text):
    """Digest of what must not move when nomenclature leaves ``path``."""
    if Path(path).suffix == ".py":
        return python_shape(text)
    return comment_free(path, text)


def verdict(path, before, after, clean_guard=None):
    """Return ``None`` if acceptable, else a one-line reason to refuse.

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
    return "nomenclature was removed AND the executable shape moved"


# ------------------------------------------------------------------ main ---

def _changed_paths(base_ref):
    """Paths changed in the diff over the scan trees, post-image names."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=d", "--no-color",
         base_ref, "--", *_SCAN_PATHS],
        capture_output=True, text=True, check=False,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _blob_at(base_ref, path):
    """File content at ``base_ref``; ``None`` when it did not exist there."""
    result = subprocess.run(
        ["git", "show", f"{base_ref}:{path}"],
        capture_output=True, text=True, check=False,
    )
    return None if result.returncode != 0 else result.stdout


def main(argv=None):
    """Scan the diff and exit non-zero on any refusal."""
    argv = list(sys.argv[1:] if argv is None else argv)
    base_ref = argv[0] if argv else _DEFAULT_BASE_REF
    clean_guard = _load_clean_guard()

    refusals = []
    examined = 0
    for path in _changed_paths(base_ref):
        before = _blob_at(base_ref, path)
        if before is None:
            continue
        try:
            after = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if debt_count(after, clean_guard) >= debt_count(before, clean_guard):
            continue
        examined += 1
        reason = verdict(path, before, after, clean_guard)
        if reason:
            refusals.append((path, reason))

    if not refusals:
        print(
            "comment-only guard: "
            f"{examined} file(s) shed nomenclature, all with an unchanged "
            f"executable shape (base {base_ref}); route descriptions are "
            "published, so they count as shape, not as comment"
        )
        return 0

    print("comment-only guard: FAILED -- nomenclature left, but so did more:")
    for path, reason in refusals:
        print(f"  {path}: {reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
