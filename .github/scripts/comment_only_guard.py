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


def _comment_free_lines(path, text):
    """``text`` as lines with every comment byte removed, trailing space cut.

    An unrecognised suffix gets no comment model at all, so every byte stays
    shape. That is the fail-closed direction: a file this guard cannot even
    tokenise is never treated as though its comments were understood.
    """
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
        names.append(getattr(node, field))
        spaces.append(space)
        setattr(node, field, "<id>")

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
        value = getattr(node, field)
        if value in maps[space]:
            setattr(node, field, maps[space][value])

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
    unjudged = []
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
        if reason is CANNOT_JUDGE:
            unjudged.append(path)
        elif is_refusal(reason):
            refusals.append((path, reason))

    # Printed before any verdict, so an absence of checking is never folded
    # into a line that reads as a check that passed.
    if unjudged:
        print("comment-only guard: NOT JUDGED -- no analyser for these files, "
              "the check belongs elsewhere:")
        for path in unjudged:
            print(f"  {path}")

    if not refusals:
        print(
            "comment-only guard: "
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
