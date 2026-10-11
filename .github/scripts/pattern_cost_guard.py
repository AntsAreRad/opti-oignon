#!/usr/bin/env python3
"""Pattern cost guard: no regular expression of the package reads a long run of text in more than linear time, unless it is owed by name.

Python's engine backtracks. A pattern whose repeats can split one run of
characters in more than one way -- two repeats side by side over classes that
share a character, a repeat inside a repeat -- or whose leading repeat is
tried again from every position of a run, takes quadratic, cubic or worse
time on a long run of such characters. A page, a tool's output or a model's
answer is long and can be steered: such a pattern is a denial of service
wherever it reads them.

WHAT IS READ. Every module of ``opti_oignon/`` (its ``data`` directory
pruned, never entered), parsed as Python reads it: a module that cannot be
parsed fails the guard by name. Every pattern given to ``compile``,
``search``, ``match``, ``fullmatch``, ``sub``, ``subn``, ``findall``,
``finditer`` or ``split`` of ``re`` -- by that name, by any name ``re`` is
imported or assigned as, or by a name one of its functions is imported or
assigned as -- with its flags. The pattern and the flags are evaluated from
constants: string and bytes literals, the names a module, class, function,
loop or comprehension binds to constants only, the items of a constant
tuple, list, set or dict a loop or a comprehension takes them from,
concatenations, ``%``, f-strings, items and slices, ``join``, ``re.escape``,
``chr`` of a constant integer, ``tuple``, ``list`` and ``sorted`` (the last
in its argument's order, not sorted: a gap known), ``re`` flags and their
union; a builtin is read as itself only where no scope binds
its name (a ``type`` alias and a type parameter included) and nothing in the
package names ``__builtins__``, ``exec`` or ``eval``, a frame's builtins or a
function's globals, imports from ``builtins`` or uses it as an object, or
reaches the ``__dict__`` of a module it imports by name at run time; a
builtin rebound by code the source shows by no name (a function made from
compiled code) is not claimed. A name is never taken for a constant when its
value can change
out of the evaluator's sight: bound by ``:=`` inside a comprehension, by a
``match`` capture or by ``import *``; rebound through ``global``,
``nonlocal``, ``globals()`` or ``vars()``, or by another module (an
attribute of an imported module stored, ``setattr``); a list, dict or set
changed in place -- by a mutating method, read or called, an item stored, an
augmented assignment -- through it or through any name that shares its
objects, or passed on, stored, returned or imported where anything could
change it. A site may so take several patterns, each measured. One built
from anything else (an argument, a configuration value, a call), one that
takes no value, one that does not compile, one whose unbounded repeat no
pump reaches, and every place ``re`` or one of its functions is read as a
value the guard cannot follow (passed, stored, called with ``**``) are
counted as dynamic and left to the contracts of their module. A bytes
pattern is read as Latin-1 text with ASCII classes.

HOW IT IS MEASURED. The pattern is parsed and, for each repeat of its tree
whose maximum is unbounded (greedy, lazy or possessive, inside groups,
branches, lookarounds, atomic groups and conditionals), pumped three ways:
its lead -- a short text the items before it on its path from the top match
-- then a run of a character its body accepts; that lead and character as a
unit repeated, which makes a search start the pattern over at each unit; and
its lead then one round of its body repeated. A back-reference stands for
the group it names. The characters are those of a fixed list the body
accepts, the ones the rest of the pattern refuses first (one it accepts lets
the match end at once), or one its own class names when it accepts none of
them. Each pump is read at doubling lengths from 2, three lengths at least,
until it takes 10 ms or reaches 16,384 characters -- 4,096 when its time
grows less than two and a half times per doubling -- with the method its
callers use: a name compiled once and only ever called with ``match`` or
``fullmatch`` from the start of its text, in its module, and never imported,
is timed so; anything else as a search. The pattern is superlinear when,
over the last two doublings of one pump, its time grew at least three times
per doubling (nine times in all) and the longest reading took at least 1 ms;
read over two doublings, a pause of the machine in one reading moves the
growth half as much. A reading that grew sixteen times or more over one
doubling and took 1 ms or more ends the pump at once, superlinear: an
exponential pattern would not finish the next length. Each time is the
least of two runs. Not claimed: a cost growing slower than the length to the
power 1.58 (three times per doubling), a blow-up that starts only after a
length the doubling reached slowly, and the patterns counted dynamic.

THE LEDGER. ``LEDGER`` names, per module, each superlinear pattern it owes by
the digest of its flags and text, as many times as it is written; it is empty
now, and a contract holds it so. It MAY
ONLY SHRINK: a superlinear pattern a module does not owe fails by its line,
digest and text, even where another owed one was made linear; an owed digest
the census no longer finds superlinear is stale and fails too, so the ledger
is lowered as patterns are made linear. A verdict against the ledger -- a
pattern not owed read superlinear or near the bar (two and a half times per
doubling or more), an owed one read linear -- is read again until two of
three readings agree, so one noisy reading neither fails a linear pattern
nor makes an owed one stale, nor lets a superlinear one through.

Witness: two planted patterns, one quadratic and one linear, are measured
first; if the first does not come out superlinear or the second does, the
guard is blind and fails. The counts read, measured, dynamic and owed are
printed on a green run.
"""
import ast
import hashlib
import itertools
import os
import re
import sys
import time
from pathlib import Path

try:
    import re._constants as sre_constants
    import re._parser as sre_parse
except ImportError:  # Python 3.10
    import sre_constants
    import sre_parse

FUNCTIONS = {"compile": 1, "search": 2, "match": 2, "fullmatch": 2, "findall": 2, "finditer": 2, "split": 3,
             "sub": 4, "subn": 4}
FIRST, SHORT, LAST, STOP, FLOOR, GROWTH, HINT, BURST = 2, 4096, 16384, 0.010, 0.001, 3.0, 2.5, 16.0
# Characters a repeat's body may accept, tried in this order: spaces first, then what patterns name most.
CANDIDATES = " \t\na0_.-:,[](){}<>=/\\'\"!?#@*+|;~" + chr(0xE9) + chr(0x3000) + chr(0x2028) + "bxZ9"
REJECT = "\x00"
PLANTED = ((r"\[\s*data\s*[:\s]\s*\w+=", re.IGNORECASE, True), (r"\[\s*data(?:\s*:\s*|\s+)\w+=", re.IGNORECASE, False))
# The values one site may take before it is counted as dynamic.
MOST = 512
# The readings of a verdict against the ledger: it counts when most of them agree.
READINGS = 3
# Among the rebound names: the package may rebind the builtins, so none is read as itself. Never a name.
BUILTINS = " builtins"

REPEATS = tuple(op for op in (sre_constants.MAX_REPEAT, sre_constants.MIN_REPEAT,
                              getattr(sre_constants, "POSSESSIVE_REPEAT", None)) if op is not None)
ATOMIC = getattr(sre_constants, "ATOMIC_GROUP", None)
ZERO_WIDTH = (sre_constants.AT, sre_constants.ASSERT, sre_constants.ASSERT_NOT)

# MAY ONLY SHRINK: the superlinear patterns each module owes, by digest. It is
# EMPTY, and a contract holds it so: every pattern the guard measures reads a
# long run in linear time, and any superlinear one fails by its line, digest
# and text, without exception.
LEDGER = {}

CATEGORIES = {
    sre_constants.CATEGORY_SPACE: r"\s", sre_constants.CATEGORY_NOT_SPACE: r"\S",
    sre_constants.CATEGORY_DIGIT: r"\d", sre_constants.CATEGORY_NOT_DIGIT: r"\D",
    sre_constants.CATEGORY_WORD: r"\w", sre_constants.CATEGORY_NOT_WORD: r"\W",
}


def digest(pattern, flags):
    """The ledger's name for a pattern: twelve hex digits of the digest of its flags and text."""
    text = pattern.decode("latin-1") if isinstance(pattern, bytes) else pattern
    return hashlib.sha256(f"{flags}:{text}".encode("utf-8", "surrogatepass")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WALKED = {}


def _walk(module):
    """Every node of a parsed module, walked once and kept while the module lives."""
    if _WALKED.get(id(module), (None,))[0] is not module:
        nodes = list(ast.walk(module))
        _WALKED[id(module)] = (module, nodes, {id(child): node for node in nodes for child in ast.iter_child_nodes(node)})
    return _WALKED[id(module)][1]


def _parents(module):
    """{id of a node: its parent} for a parsed module."""
    _walk(module)
    return _WALKED[id(module)][2]


class Unknown(Exception):
    """A value the guard cannot know from the source alone."""


def _product(options, combine):
    out = []
    for parts in itertools.product(*options):
        value = combine(parts)
        if value not in out:
            out.append(value)
        if len(out) > MOST:
            raise Unknown
    return out


def _items(value):
    """The items a loop takes from a constant: a container's, or a string's characters."""
    if isinstance(value, (dict, tuple, list, set, frozenset, str)):
        return tuple(value)
    raise Unknown


def _index(base, key):
    """``base[key]`` for constants, or Unknown."""
    try:
        return base[key]
    except (TypeError, KeyError, IndexError, ValueError):
        raise Unknown from None


def _part(item, path):
    for position in path:
        if not isinstance(item, tuple) or position >= len(item):
            raise Unknown
        item = item[position]
    return item


COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
MUTABLE = (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)
MUTATORS = {"append", "extend", "insert", "update", "add", "setdefault", "pop", "popitem", "remove", "discard", "clear",
            "sort", "reverse"}


def _roots(node):
    """The names a value may share objects with: the name itself, what it is read from, what it is built of."""
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Subscript, ast.Attribute, ast.Starred, ast.NamedExpr)):
        return _roots(node.value)
    parts = []
    if isinstance(node, ast.Call):
        parts = ([node.func.value] if isinstance(node.func, ast.Attribute) else []) + node.args + [
            keyword.value for keyword in node.keywords]
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        parts = node.elts
    elif isinstance(node, ast.Dict):
        parts = [value for value in node.values]
    elif isinstance(node, ast.BinOp):
        parts = [node.left, node.right]
    elif isinstance(node, ast.BoolOp):
        parts = node.values
    elif isinstance(node, ast.IfExp):
        parts = [node.body, node.orelse]
    elif isinstance(node, COMPREHENSIONS):
        parts = [generator.iter for generator in node.generators]
    return set().union(*map(_roots, parts)) if parts else set()


def _is_namespace(node):
    """True for ``globals()`` or ``vars()``: the module's own names as a dict."""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("globals", "vars")
            and not node.args)


def mutated(module):
    """The names the module changes in place or rebinds from another scope: a mutating method read on the name, an
    item or slice of it stored or deleted, an augmented assignment, ``global`` or ``nonlocal``, and every name that
    shares objects with one of those (bound from it, taken from it in a loop); a name stored through ``globals()``
    or ``vars()``; every name, when the module calls ``exec`` or ``eval``. Their values are never known; ``"*"`` among them: the module's names may all be rebound. A
    change in place counts for the names bound to a list, dict or set: a string or a number never changes."""
    out, rebound_here, edges, containers = set(), set(), [], set()
    for node in _walk(module):
        if isinstance(node, ast.Name) and node.id in ("exec", "eval"):
            rebound_here.add("*")
        if isinstance(node, ast.Attribute) and node.attr in MUTATORS:
            if _is_namespace(node.value):
                rebound_here.add("*")
            else:
                out |= _roots(node.value)
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if _is_namespace(node.value):
                key = node.slice
                rebound_here.add(key.value if isinstance(key, ast.Constant) and isinstance(key.value, str) else "*")
            else:
                out |= _roots(node.value)
        elif isinstance(node, ast.AugAssign):
            out |= _roots(node.target)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            rebound_here.update(node.names)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and (isinstance(node.value, MUTABLE) or (
                isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id in ("list", "dict", "set", "sorted", "bytearray"))):
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                containers.update(name for name, _path in _paths(target))
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                edges += [(name, _roots(node.value)) for name, _path in _paths(target)]
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            edges += [(name, _roots(node.iter)) for name, _path in _paths(node.target)]
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            edges += [(name, _roots(item.context_expr)) for item in node.items if item.optional_vars is not None
                      for name, _path in _paths(item.optional_vars)]
    grown = True
    while grown:
        grown = False
        for name, roots in edges:
            if name in out and not roots <= out:
                out |= roots
                grown = True
    return (out & containers) | rebound_here


def rebound(parsed):
    """The names a module of the package rebinds in another: an attribute of a module it imports, or of one it binds
    from ``__import__`` or ``import_module``, stored or deleted, or set by ``setattr`` or ``delattr`` under a constant
    name; ``"*"`` when one is set under a name not known; ``BUILTINS`` when a module names ``__builtins__``, ``exec``
    or ``eval``, a frame's ``f_builtins`` or a function's ``__globals__``, imports from ``builtins``, or uses an
    imported ``builtins`` as an object rather than reading one of its names, any of which can rebind a builtin."""
    out = set()
    for _rel, module in parsed:
        modules = {(alias.asname or alias.name).split(".")[0] for node in _walk(module)
                   if isinstance(node, ast.Import) for alias in node.names}
        modules |= {name for node in _walk(module) if isinstance(node, ast.Assign) and _importer(node.value)
                    for target in node.targets for name, _path in _paths(target)}
        spaces = {alias.asname or alias.name for node in _walk(module) if isinstance(node, ast.Import)
                  for alias in node.names if alias.name == "builtins"}
        dynamic = modules - {(alias.asname or alias.name).split(".")[0] for node in _walk(module)
                             if isinstance(node, ast.Import) for alias in node.names}
        parents = _parents(module)
        for node in _walk(module):
            parent = parents.get(id(node))
            read = isinstance(parent, ast.Attribute) and parent.value is node and parent.attr != "__dict__"
            if (isinstance(node, ast.Name) and (node.id in ("__builtins__", "exec", "eval") or node.id in spaces and not read)
                    or isinstance(node, ast.Attribute) and (
                        node.attr in ("f_builtins", "__globals__", "__builtins__")
                        or node.attr == "__dict__" and (_importer(node.value) or _roots(node.value) & dynamic))
                    or isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "vars"
                    and node.args and _roots(node.args[0]) & dynamic
                    or isinstance(node, ast.ImportFrom) and node.module == "builtins"):
                out.add(BUILTINS)
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
                if _roots(node.value) & modules or _importer(node.value):
                    out.add(node.attr)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("setattr", "delattr")
                  and len(node.args) >= 2 and (_roots(node.args[0]) & modules or _importer(node.args[0]))):
                name = node.args[1]
                out.add(name.value if isinstance(name, ast.Constant) and isinstance(name.value, str) else "*")
    return out


def _importer(node):
    """True for a call that returns a module by its name: ``__import__(...)``, ``import_module(...)``."""
    return isinstance(node, ast.Call) and (
        isinstance(node.func, ast.Name) and node.func.id in ("__import__", "import_module")
        or isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")


PURE = {"len", "tuple", "list", "sorted", "set", "frozenset", "enumerate", "zip", "any", "all", "min", "max", "sum",
        "reversed", "iter", "map", "filter", "str", "repr", "bool", "dict"}
READERS = {"get", "items", "keys", "values", "copy", "index", "count"}


def escaped(module, imported):
    """The names bound to a list, dict or set somewhere in the module that something could change out of the
    module's sight: read anywhere but by a loop, ``in``, an index, an operator, a pure builtin, ``join`` or a
    reading method, or imported by another module."""
    mutable = {name for node in _walk(module) if isinstance(node, (ast.Assign, ast.AnnAssign))
               and isinstance(node.value, MUTABLE)
               for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
               for name, _path in _paths(target)}
    parents = _parents(module)
    out = {name for name in mutable if name in imported}
    for node in _walk(module):
        if not (isinstance(node, ast.Name) and node.id in mutable and isinstance(node.ctx, ast.Load)):
            continue
        parent = parents.get(id(node))
        safe = (isinstance(parent, (ast.For, ast.AsyncFor, ast.comprehension)) and parent.iter is node
                or isinstance(parent, ast.Compare) and node in parent.comparators
                and all(isinstance(op, (ast.In, ast.NotIn)) for op in parent.ops)
                or isinstance(parent, ast.Subscript) and parent.value is node and isinstance(parent.ctx, ast.Load)
                or isinstance(parent, ast.Call) and node in parent.args and (
                    isinstance(parent.func, ast.Name) and parent.func.id in PURE
                    or isinstance(parent.func, ast.Attribute) and parent.func.attr == "join")
                or isinstance(parent, ast.Attribute) and parent.attr in READERS | MUTATORS
                or isinstance(parent, (ast.Starred, ast.BinOp)))
        if not safe:
            out.add(node.id)
    return out


class Scope:
    """The values a name can hold in one module, class or function: those of every binding of it there when each
    is a constant, the enclosing scope's when it is bound nowhere there; never those of a name changed in place."""

    def __init__(self, bindings, parent, re_names, changed=frozenset()):
        self.bindings, self.parent, self.re_names = bindings, parent, re_names
        self.changed = parent.changed if parent is not None else changed
        self.memo, self.busy = {}, set()

    def lookup(self, name):
        if name in self.changed or "*" in self.changed or "*" in self.bindings:
            raise Unknown
        if name not in self.bindings:
            if self.parent is None:
                raise Unknown
            return self.parent.lookup(name)
        if name not in self.memo:
            if name in self.busy:
                raise Unknown
            self.busy.add(name)
            try:
                found = []
                for path, node in self.bindings[name]:
                    if path is None:
                        raise Unknown
                    values = self.values(node)
                    if path:
                        values = [_part(item, path[1:]) for value in values for item in _items(value)]
                    found += [value for value in values if value not in found]
                    if len(found) > MOST:
                        raise Unknown
                self.memo[name] = found
            except Unknown:
                self.memo[name] = None
            finally:
                self.busy.discard(name)
        if self.memo[name] is None:
            raise Unknown
        return self.memo[name]

    def values(self, node):
        """Every value ``node`` can take, or Unknown."""
        if isinstance(node, ast.Constant):
            return [node.value]
        if isinstance(node, ast.Name):
            return self.lookup(node.id)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return _product([self.values(item) for item in node.elts], tuple)
        if isinstance(node, ast.Dict) and None not in node.keys:
            pairs = _product([self.values(key) for key in node.keys] + [self.values(value) for value in node.values],
                             tuple)
            half = len(node.keys)
            return [dict(zip(pair[:half], pair[half:])) for pair in pairs]
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod, ast.BitOr)):
            combine = {ast.Add: lambda p: p[0] + p[1], ast.Mod: lambda p: p[0] % p[1],
                       ast.BitOr: lambda p: p[0] | p[1]}[type(node.op)]
            try:
                return _product([self.values(node.left), self.values(node.right)], combine)
            except (TypeError, ValueError):
                raise Unknown from None
        if isinstance(node, ast.JoinedStr):
            parts = []
            for part in node.values:
                if isinstance(part, ast.Constant):
                    parts.append([part.value])
                elif isinstance(part, ast.FormattedValue) and part.conversion == -1 and part.format_spec is None:
                    values = self.values(part.value)
                    if not all(isinstance(value, (str, int)) and not isinstance(value, bool) for value in values):
                        raise Unknown
                    parts.append([str(value) for value in values])
                else:
                    raise Unknown
            return _product(parts, "".join)
        if isinstance(node, ast.Subscript):
            index = node.slice
            if isinstance(index, ast.Slice):
                bounds = [[None] if part is None else self.values(part) for part in (index.lower, index.upper, index.step)]
                keys = [slice(*part) for part in itertools.product(*bounds)]
            else:
                keys = self.values(index)
            return _product([self.values(node.value), keys], lambda pair: _index(*pair))
        if isinstance(node, ast.IfExp):
            body = self.values(node.body)
            return body + [value for value in self.values(node.orelse) if value not in body]
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in self.re_names:
            value = getattr(re, node.attr, None)
            if isinstance(value, int):
                return [int(value)]
        if isinstance(node, ast.Call) and not node.keywords:
            return self._call(node)
        raise Unknown

    def _call(self, node):
        func, args = node.func, node.args
        if isinstance(func, ast.Attribute) and len(args) <= 1:
            owner, method = func.value, func.attr
            if isinstance(owner, ast.Name) and owner.id in self.re_names and method == "escape" and len(args) == 1:
                return [re.escape(value) for value in self.values(args[0]) if isinstance(value, (str, bytes))]
            if method == "join" and len(args) == 1:
                try:
                    return _product([self.values(owner), self.values(args[0])], lambda p: p[0].join(_items(p[1])))
                except (TypeError, AttributeError):
                    raise Unknown from None
            if method in ("items", "keys", "values") and not args:
                return [tuple(getattr(value, method)()) for value in self.values(owner) if isinstance(value, dict)]
        if isinstance(func, ast.Name) and len(args) == 1 and self.builtin(func.id):
            if func.id in ("tuple", "list", "sorted"):
                return [tuple(_items(value)) for value in self.values(args[0])]
            if func.id == "chr":
                values = self.values(args[0])
                if not all(type(value) is int and 0 <= value <= sys.maxunicode for value in values):
                    raise Unknown
                return [chr(value) for value in values]
        raise Unknown

    def builtin(self, name):
        """True when ``name`` is read as the builtin of that name here: no scope up to the module binds it, and
        nothing rebinds it out of sight."""
        if name in self.changed or "*" in self.changed or BUILTINS in self.changed:
            return False
        scope = self
        while scope is not None:
            if name in scope.bindings or "*" in scope.bindings:
                return False
            scope = scope.parent
        return True


def _paths(target, path=()):
    """(name, path) for each name a target binds; the path is the positions of the tuples it is unpacked from."""
    if isinstance(target, ast.Name):
        yield target.id, path
    elif isinstance(target, ast.Starred):
        yield from ((name, None) for name, _path in _paths(target.value))
    elif isinstance(target, (ast.Tuple, ast.List)):
        for position, item in enumerate(target.elts):
            yield from _paths(item, None if path is None else path + (position,))


MATCH_CAPTURES = tuple(getattr(ast, kind) for kind in ("MatchAs", "MatchStar", "MatchMapping") if hasattr(ast, kind))
TYPE_ALIASES = tuple(getattr(ast, kind) for kind in ("TypeAlias",) if hasattr(ast, kind))


def bindings(body, params=(), generators=()):
    """name -> [(path, node)]: ``()`` bound to the node's value, ``(0, ...)`` to an item of it (then the positions it
    is unpacked from), None to something unknown. Nested functions and classes are bound, not entered; the targets
    of a comprehension belong to its own scope (``generators``), the names its ``:=`` binds to the enclosing one."""
    out = {name: [(None, None)] for name in params}

    def bind(target, node, each):
        for name, path in _paths(target):
            kind = None if path is None else (0,) + path if each else path if not path else None
            out.setdefault(name, []).append((kind, node))

    def visit(node, comprehended=False):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.setdefault(node.name, []).append((None, node))
            return
        if isinstance(node, TYPE_ALIASES):
            out.setdefault(node.name.id, []).append((None, node))
            return
        if isinstance(node, ast.Lambda):
            return
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind(target, node.value, False)
        elif isinstance(node, ast.NamedExpr) and comprehended:
            # Its value reads the comprehension's own names: not known here.
            out.setdefault(node.target.id, []).append((None, node))
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            bind(node.target, node.value, False)
        elif isinstance(node, MATCH_CAPTURES):
            for name in (getattr(node, "name", None), getattr(node, "rest", None)):
                if name:
                    out.setdefault(name, []).append((None, node))
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            bind(node.target, node.iter, True)
        elif isinstance(node, (ast.AugAssign, ast.With, ast.AsyncWith, ast.Import, ast.ImportFrom, ast.ExceptHandler,
                               ast.Global, ast.Nonlocal)):
            names = []
            if isinstance(node, ast.AugAssign):
                names = [name for name, _path in _paths(node.target)]
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                names = [name for item in node.items if item.optional_vars is not None
                         for name, _path in _paths(item.optional_vars)]
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [(alias.asname or alias.name).split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ExceptHandler):
                names = [node.name] if node.name else []
            else:
                names = list(node.names)
            for name in names:
                out.setdefault(name, []).append((None, node))
        for child in ast.iter_child_nodes(node):
            visit(child, comprehended or isinstance(node, COMPREHENSIONS))

    for generator in generators:
        bind(generator.target, generator.iter, True)
    for node in body:
        visit(node)
    return out


def _params(arguments):
    return [arg.arg for arg in arguments.posonlyargs + arguments.args + arguments.kwonlyargs] + [
        arg.arg for arg in (arguments.vararg, arguments.kwarg) if arg is not None]


def _type_params(node):
    """The names a function or class binds as type parameters (``def f[T]()``): bound in its own scope."""
    return [param.name for param in getattr(node, "type_params", ())]


def scoped_calls(module, re_names, imported=frozenset(), elsewhere=frozenset()):
    """(call, scope) for every call of the module, the scope being where its names are read."""
    found = []

    def visit(node, scope, outer):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for child in getattr(node, "decorator_list", []) + node.args.defaults + [
                    default for default in node.args.kw_defaults if default is not None]:
                visit(child, scope, outer)
            body = node.body if isinstance(node.body, list) else [node.body]
            inner = Scope(bindings(body, _params(node.args) + _type_params(node)), outer, re_names)
            for child in body:
                visit(child, inner, inner)
            return
        if isinstance(node, ast.ClassDef):
            for child in node.decorator_list + node.bases + [keyword.value for keyword in node.keywords]:
                visit(child, scope, outer)
            inner = Scope(bindings(node.body, _type_params(node)), scope, re_names)
            for child in node.body:
                visit(child, inner, outer)
            return
        if isinstance(node, (ast.For, ast.AsyncFor)):
            names = {name for name, _path in _paths(node.target)}
            inside = bindings(node.body + node.orelse)
            loop = bindings([], generators=[node])
            loop.update({name: loop.get(name, []) + inside[name] for name in names if name in inside})
            visit(node.iter, scope, outer)
            inner = Scope(loop, scope, re_names)
            for child in node.body + node.orelse:
                visit(child, inner, outer)
            return
        if isinstance(node, COMPREHENSIONS):
            inner = Scope(bindings([], generators=node.generators), scope, re_names)
            visit(node.generators[0].iter, scope, outer)
            parts = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
            for child in parts + node.generators[0].ifs + [part for generator in node.generators[1:]
                                                            for part in [generator.iter] + generator.ifs]:
                visit(child, inner, outer)
            return
        if isinstance(node, ast.Call):
            found.append((node, scope))
        for child in ast.iter_child_nodes(node):
            visit(child, scope, outer)

    top = Scope(bindings(module.body), None, re_names, mutated(module) | escaped(module, imported) | elsewhere)
    for node in module.body:
        visit(node, top, top)
    return found


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------
ANCHORED = {"match", "fullmatch"}
# The methods of a module the guard could not parse.
UNREAD = ("unread",)


def _name_of(node):
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None


def _from_start(call):
    """True when a ``match`` or ``fullmatch`` call reads from the start of its text: no position, or 0."""
    position = call.args[1] if len(call.args) > 1 else next(
        (keyword.value for keyword in call.keywords if keyword.arg == "pos"), None)
    return position is None or (isinstance(position, ast.Constant) and position.value == 0)


def _modes(module, imported):
    """For each name a pattern is compiled into in the module, the methods it is timed with.

    Those of ``match`` and ``fullmatch`` called on it by that name from the start of the text, when every read of
    the name in the module is such a call and no module of the package imports the name or reads it from this
    one; ``search`` otherwise -- a name put in a container, passed, returned, imported or matched from a position
    is not followed, and is timed in the mode that costs most.
    """
    calls, reads = {}, {}
    owners = set()
    for node in _walk(module):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            name = _name_of(node.func.value)
            if name is not None:
                anchored = node.func.attr in ANCHORED and _from_start(node)
                calls.setdefault(name, set()).add(node.func.attr if anchored else "search")
                owners.add(id(node.func.value))
    for node in _walk(module):
        name = _name_of(node) if isinstance(getattr(node, "ctx", None), ast.Load) else None
        if name is not None:
            reads.setdefault(name, []).append(id(node))
    modes = {}
    for name, methods in calls.items():
        followed = all(read in owners for read in reads.get(name, ())) and name not in imported
        modes[name] = tuple(sorted(methods)) if followed and methods <= ANCHORED else ("search",)
    return modes


def _imported(parsed):
    """Every name a module of the package imports from another or reads as an attribute of one it imports."""
    names = set()
    for _rel, module in parsed:
        modules = set()
        for node in _walk(module):
            if isinstance(node, ast.ImportFrom):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                modules.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        for node in _walk(module):
            if isinstance(node, ast.Attribute):
                root = node.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name) and root.id in modules:
                    names.add(node.attr)
    return names


def _re_names(module):
    """(names ``re`` is bound to, {local name: function} for its functions): imported so, or assigned from one."""
    aliases, direct = {"re"}, {}
    for node in _walk(module):
        if isinstance(node, ast.Import):
            aliases.update(alias.asname or alias.name for alias in node.names if alias.name == "re")
        elif isinstance(node, ast.ImportFrom) and node.module == "re" and not node.level:
            direct.update({alias.asname or alias.name: alias.name for alias in node.names if alias.name in FUNCTIONS})
    assigned = [(node.targets[0].id, node.value) for node in _walk(module) if isinstance(node, ast.Assign)
                and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)]
    grown = True
    while grown:
        grown = False
        for name, value in assigned:
            if isinstance(value, ast.Name) and value.id in aliases and name not in aliases:
                aliases.add(name)
                grown = True
            function = (value.attr if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name)
                        and value.value.id in aliases and value.attr in FUNCTIONS
                        else direct.get(value.id) if isinstance(value, ast.Name) else None)
            if function and name not in direct:
                direct[name] = function
                grown = True
    return aliases, direct


def _loose(module, aliases, direct):
    """The places ``re`` or one of its functions is read as a value the guard cannot follow -- passed, stored, put
    in a container -- each a site whose pattern is unknown."""
    parents = _parents(module)
    out = []
    for node in _walk(module):
        if not isinstance(getattr(node, "ctx", None), ast.Load):
            continue
        parent = parents.get(id(node))
        followed_alias = (isinstance(parent, ast.Assign) and parent.value is node and len(parent.targets) == 1
                          and isinstance(parent.targets[0], ast.Name))
        if isinstance(node, ast.Name) and node.id in aliases:
            if not (isinstance(parent, ast.Attribute) and parent.value is node) and not followed_alias:
                out.append(node)
        elif (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases
              and node.attr in FUNCTIONS or isinstance(node, ast.Name) and node.id in direct):
            if not (isinstance(parent, ast.Call) and parent.func is node) and not followed_alias:
                out.append(node)
    return out


def _package(root):
    """(relative path, parsed module or None when it cannot be parsed) for every module of the package, read as
    Python reads it (its coding declaration honoured), its own data directory pruned."""
    base = root / "opti_oignon"
    parsed = []
    # Walked with the package's data directory pruned, never entered: it holds a user's own files.
    for folder, dirs, files in os.walk(base):
        here = Path(folder)
        dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not (here == base and d == "data"))
        for name in sorted(files):
            if name.endswith(".py"):
                path = here / name
                try:
                    parsed.append((path.relative_to(root), ast.parse(path.read_bytes())))
                except (SyntaxError, ValueError):
                    parsed.append((path.relative_to(root), None))
    return parsed


def sites(root):
    """Every pattern a site of the package can take: (module, line, column, pattern or None, flags or None, methods).

    A site whose pattern or flags cannot be evaluated gives one row of None. The methods are the ones a search is
    timed with: the anchored function of a direct call (``match``, ``fullmatch``), or for a pattern compiled into
    a name, those ``_modes`` reads; ``search`` otherwise.
    """
    found = []
    parsed = _package(root)
    for rel, module in parsed:
        if module is None:
            found.append((rel.as_posix(), 0, 0, None, None, UNREAD))
    parsed = [(rel, module) for rel, module in parsed if module is not None]
    imported, elsewhere = _imported(parsed), rebound(parsed)
    for rel, module in parsed:
        modes = _modes(module, imported)
        aliases, direct = _re_names(module)
        for node in _loose(module, aliases, direct):
            found.append((rel.as_posix(), node.lineno, node.col_offset, None, None, ("search",)))
        bound = {}
        for node in _walk(module):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.value, ast.Call):
                name = _name_of(node.targets[0])
                if name is not None:
                    bound[id(node.value)] = name
        for node, scope in scoped_calls(module, aliases, imported, elsewhere):
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in aliases and node.func.attr in FUNCTIONS):
                function = node.func.attr
            elif isinstance(node.func, ast.Name) and node.func.id in direct:
                function = direct[node.func.id]
            else:
                continue
            first = node.args[0] if node.args and not isinstance(node.args[0], ast.Starred) else next(
                (keyword.value for keyword in node.keywords if keyword.arg == "pattern"), None)
            if first is None:
                found.append((rel.as_posix(), node.lineno, node.col_offset, None, None, ("search",)))
                continue
            position = FUNCTIONS[function]
            flag_node = node.args[position] if len(node.args) > position else None
            for keyword in node.keywords:
                if keyword.arg == "flags":
                    flag_node = keyword.value
            if function in ANCHORED:
                methods = (function,)
            elif function == "compile" and id(node) in bound:
                methods = modes.get(bound[id(node)], ("search",))
            else:
                methods = ("search",)
            try:
                patterns = scope.values(first)
                flags = [0] if flag_node is None else scope.values(flag_node)
                if not patterns or not flags or not all(isinstance(p, (str, bytes)) for p in patterns) or not all(
                        isinstance(f, int) and not isinstance(f, bool) for f in flags):
                    raise Unknown
                pairs = _product([patterns, flags], tuple)
            except Unknown:
                found.append((rel.as_posix(), node.lineno, node.col_offset, None, None, methods))
                continue
            for pattern, flag in pairs:
                found.append((rel.as_posix(), node.lineno, node.col_offset, pattern, flag, methods))
    return found


# ---------------------------------------------------------------------------
# Pumps
# ---------------------------------------------------------------------------
def _one(item):
    """A pattern for one single-character item of a parsed tree."""
    op, av = item
    if op is sre_constants.LITERAL:
        return re.escape(chr(av))
    if op is sre_constants.NOT_LITERAL:
        return "[^" + re.escape(chr(av)) + "]"
    if op is sre_constants.ANY:
        return "."
    if op is sre_constants.IN:
        parts, negate = [], ""
        for sub_op, sub_av in av:
            if sub_op is sre_constants.NEGATE:
                negate = "^"
            elif sub_op is sre_constants.LITERAL:
                parts.append(re.escape(chr(sub_av)))
            elif sub_op is sre_constants.RANGE:
                parts.append(re.escape(chr(sub_av[0])) + "-" + re.escape(chr(sub_av[1])))
            elif sub_op is sre_constants.CATEGORY:
                parts.append(CATEGORIES.get(sub_av, ""))
        return "[" + negate + "".join(parts) + "]" if parts or negate else None
    return None


def _named(item):
    """The characters a single-character item names itself: its literal, the members and range starts of its class."""
    op, av = item
    members = av if op is sre_constants.IN else [(op, av)]
    return [chr(sub_av) if sub_op is sre_constants.LITERAL else chr(sub_av[0]) for sub_op, sub_av in members
            if sub_op in (sre_constants.LITERAL, sre_constants.RANGE)]


def first_chars(seq, flags, groups=None):
    """The characters the first single-character item of ``seq`` accepts: those of the candidates it accepts, or
    else the first it names itself that it accepts; a back-reference stands for the group it names."""
    for op, av in seq:
        if op in (sre_constants.LITERAL, sre_constants.NOT_LITERAL, sre_constants.ANY, sre_constants.IN):
            single = _one((op, av))
            if single is None:
                return []
            item = re.compile(single, flags)
            accepted = [ch for ch in CANDIDATES if item.fullmatch(ch)]
            return accepted or [ch for ch in _named((op, av)) if item.fullmatch(ch)][:1]
        if op in REPEATS:
            return first_chars(av[2], flags, groups)
        if op is sre_constants.SUBPATTERN:
            return first_chars(av[-1], flags, groups)
        if op is ATOMIC:
            return first_chars(av, flags, groups)
        if op is sre_constants.GROUPREF:
            return first_chars((groups or {}).get(av, []), flags, groups)
        if op is sre_constants.BRANCH or op is sre_constants.GROUPREF_EXISTS:
            out = []
            for branch in (av[1] if op is sre_constants.BRANCH else [b for b in av[1:] if b is not None]):
                out += [ch for ch in first_chars(branch, flags, groups) if ch not in out]
            return out
        if op in ZERO_WIDTH:
            continue
        return []
    return []


def _sample(seq, flags, groups=None):
    """A short text the parsed ``seq`` matches -- a class by the first character it accepts, a repeat at its least
    and once at least, a branch by the first that can be written -- or None when one of its items cannot be
    written."""
    out = []
    for op, av in seq:
        if op in (sre_constants.LITERAL, sre_constants.NOT_LITERAL, sre_constants.ANY, sre_constants.IN):
            accepted = first_chars([(op, av)], flags)
            if not accepted:
                return None
            out.append(accepted[0])
        elif op is sre_constants.GROUPREF:
            inner = _sample(groups[av], flags, groups) if groups and av in groups else None
            if inner is None:
                return None
            out.append(inner)
        elif op in REPEATS:
            body = _sample(av[2], flags, groups)
            if body is None:
                return None
            out.append(body * max(av[0], 1))
        elif op is sre_constants.SUBPATTERN or op is ATOMIC:
            inner = _sample(av if op is ATOMIC else av[-1], flags, groups)
            if inner is None:
                return None
            out.append(inner)
        elif op is sre_constants.BRANCH or op is sre_constants.GROUPREF_EXISTS:
            branches = av[1] if op is sre_constants.BRANCH else [b if b is not None else [] for b in av[1:]]
            inner = next((s for s in (_sample(b, flags, groups) for b in branches) if s is not None), None)
            if inner is None:
                return None
            out.append(inner)
        elif op not in ZERO_WIDTH:
            return None
    return "".join(out)


def _lead(seq, upto, flags, groups=None):
    """A short text the items of ``seq`` before ``upto`` match, each written when it can be, left out when not."""
    out = ""
    for item in seq[:upto]:
        if item[0] not in ZERO_WIDTH:
            out += _sample([item], flags, groups) or ""
    return out


def _groups(seq, out=None):
    """{group number: its parsed items} for every capturing group of a parsed tree."""
    out = {} if out is None else out
    for op, av in seq:
        if op is sre_constants.SUBPATTERN:
            if av[0] is not None:
                out[av[0]] = av[-1]
            _groups(av[-1], out)
        elif op in REPEATS:
            _groups(av[2], out)
        elif op is ATOMIC:
            _groups(av, out)
        elif op in (sre_constants.ASSERT, sre_constants.ASSERT_NOT):
            _groups(av[1], out)
        elif op is sre_constants.BRANCH:
            for branch in av[1]:
                _groups(branch, out)
        elif op is sre_constants.GROUPREF_EXISTS:
            for branch in av[1:]:
                if branch is not None:
                    _groups(branch, out)
    return out


def unbounded(pattern, flags):
    """True when the parsed pattern holds a repeat with no maximum."""
    def walk(seq):
        for op, av in seq:
            if op in REPEATS and (av[1] == sre_constants.MAXREPEAT or walk(av[2])):
                return True
            inner = (av[-1] if op is sre_constants.SUBPATTERN else av if op is ATOMIC
                     else av[1] if op in (sre_constants.ASSERT, sre_constants.ASSERT_NOT) else None)
            if inner is not None and walk(inner):
                return True
            branches = av[1] if op is sre_constants.BRANCH else av[1:] if op is sre_constants.GROUPREF_EXISTS else []
            if any(branch is not None and walk(branch) for branch in branches):
                return True
        return False
    return walk(list(sre_parse.parse(pattern, flags)))


def pumps(pattern, flags):
    """(lead, character, body) for each unbounded repeat of the pattern and each of up to four characters its body
    accepts; ``body`` is a text one round of the repeat matches, when it is longer than one character.

    The lead is a text the items before the repeat on its path from the top match: the parser factors a prefix
    the branches share out of them, so a branch alone does not hold it.
    """
    found = []
    tree = list(sre_parse.parse(pattern, flags))
    groups = _groups(tree)

    def walk(seq, outer, after):
        for index, (op, av) in enumerate(seq):
            here = outer + _lead(seq, index, flags, groups)
            follow = first_chars(seq[index + 1:], flags, groups) if index + 1 < len(seq) else after
            if op in REPEATS:
                if av[1] == sre_constants.MAXREPEAT:
                    body = _sample(av[2], flags, groups)
                    body = body if body is not None and len(body) > 1 else None
                    accepted = first_chars(av[2], flags, groups)
                    # A character the continuation refuses first: one it accepts lets the match end at once.
                    accepted = [ch for ch in accepted if ch not in follow] + [ch for ch in accepted if ch in follow]
                    for ch in accepted[:4]:
                        if (here, ch, body) not in found:
                            found.append((here, ch, body))
                walk(av[2], here, follow)
            elif op is sre_constants.SUBPATTERN:
                walk(av[-1], here, follow)
            elif op is ATOMIC:
                walk(av, here, follow)
            elif op is sre_constants.BRANCH:
                for branch in av[1]:
                    walk(branch, here, follow)
            elif op is sre_constants.GROUPREF_EXISTS:
                for branch in av[1:]:
                    if branch is not None:
                        walk(branch, here, follow)
            elif op in (sre_constants.ASSERT, sre_constants.ASSERT_NOT):
                walk(av[1], here, [])

    walk(tree, "", [])
    return found


def _timed(compiled, text, methods):
    """The least of two runs of the costliest of ``methods`` on ``text``."""
    worst = 0.0
    for method in methods:
        read = getattr(compiled, method)
        best = None
        for _ in range(2):
            start = time.perf_counter()
            read(text)
            spent = time.perf_counter() - start
            best = spent if best is None else min(best, spent)
        worst = max(worst, best)
    return worst


def _texts(lead, ch, body, n):
    """The pumps of a repeat, about ``n`` characters each: its lead then a run of ``ch``; the unit lead-and-``ch``
    repeated, which makes a search start the pattern over at each unit; and its lead then one round of its body
    repeated, a word or a tag where a run of one character would not split the same way."""
    texts = {"run": lead + ch * n + REJECT}
    if lead:
        unit = lead + ch
        texts["unit"] = unit * max(1, n // len(unit)) + REJECT
    if body:
        texts["body"] = lead + body * max(1, n // len(body)) + REJECT
    return texts


def _as_text(pattern, flags):
    """A bytes pattern as the text pattern it is measured as: Latin-1, ASCII classes, no locale."""
    if isinstance(pattern, bytes):
        return pattern.decode("latin-1"), (flags & ~re.LOCALE) | re.ASCII
    return pattern, flags


def growth(pattern, flags, methods=("search",)):
    """The first superlinear pump of a pattern read with ``methods``, else its worst: (growth ratio, seconds at the
    longer run, pump, character)."""
    pattern, flags = _as_text(pattern, flags)
    compiled = re.compile(pattern, flags)
    worst = None
    for lead, ch, body in pumps(pattern, flags):
        for kind in ("run", "unit", "body"):
            if kind not in _texts(lead, ch, body, 1):
                continue
            label = {"run": lead, "unit": f"({lead}{ch})*", "body": f"{lead}({body})*"}[kind]
            times, n = [], FIRST
            while True:
                text = _texts(lead, ch, body, n)[kind]
                spent = _timed(compiled, text, methods)
                times.append(spent)
                # A burst -- sixteen times or more over one doubling -- would not finish the next length: it ends
                # the pump at once, superlinear.
                step = spent / times[-2] if len(times) >= 2 and times[-2] > 0 else 1.0
                if step >= BURST and spent >= FLOOR:
                    return step, spent, label, ch
                # The growth per doubling over the last two: a pause of the machine in one reading moves it half as
                # much as it would move one doubling.
                ratio = (spent / times[-3]) ** 0.5 if len(times) >= 3 and times[-3] > 0 else 1.0
                if len(times) >= 3 and (spent >= STOP or n >= LAST or (n >= SHORT and ratio < HINT)):
                    if superlinear((ratio, spent)):
                        return ratio, spent, label, ch
                    if worst is None or spent > worst[1]:
                        worst = (ratio, spent, label, ch)
                    break
                n *= 2
    return worst


def superlinear(result):
    return result is not None and result[1] >= FLOOR and result[0] >= GROWTH


def steady(pattern, flags, methods=("search",), *taken):
    """(verdict, reading): the verdict most of READINGS readings of the pattern agree on, the readings ``taken``
    counted first, and one reading that gives it. The guard reads a verdict against the ledger so."""
    readings = list(taken)
    while True:
        steep = [reading for reading in readings if superlinear(reading)]
        flat = [reading for reading in readings if not superlinear(reading)]
        if len(steep) * 2 > READINGS or len(flat) * 2 > READINGS:
            return (True, steep[0]) if len(steep) * 2 > READINGS else (False, flat[0])
        readings.append(growth(pattern, flags, methods))


# ---------------------------------------------------------------------------
# Census and ledger
# ---------------------------------------------------------------------------
def census(root):
    """(rows, dynamic, read, unread): each measured pattern (module, line, pattern, flags, methods, result); the
    sites whose pattern is not a constant, does not compile or holds an unbounded repeat no pump reaches; the count
    of sites read; the modules that could not be parsed."""
    rows, dynamic, read, unread, measured = [], [], set(), [], {}
    for module, line, column, pattern, flags, methods in sites(root):
        if methods == UNREAD:
            unread.append(module)
            continue
        read.add((module, line, column))
        if pattern is None:
            dynamic.append((module, line))
            continue
        key = (pattern, flags, methods)
        if key not in measured:
            try:
                result = growth(pattern, flags, methods)
                blind = result is None and unbounded(*_as_text(pattern, flags))
                measured[key] = Unknown if blind else result
            except (re.error, RecursionError, OverflowError, ValueError, TypeError):
                measured[key] = Unknown
        if measured[key] is Unknown:
            dynamic.append((module, line))
        else:
            rows.append((module, line, pattern, flags, methods, measured[key]))
    return rows, dynamic, len(read), unread


def _shown(pattern):
    return ascii(pattern.decode("latin-1") if isinstance(pattern, bytes) else pattern)[1:-1][:80]


def run(root):
    """(code, lines): 0 and one green line, or 1 and each finding by name."""
    lines = []
    for pattern, flags, quadratic in PLANTED:
        if superlinear(growth(pattern, flags)) != quadratic:
            return 1, [f"Pattern cost: FAILED -- blind: the planted {'quadratic' if quadratic else 'linear'} pattern "
                       f"{pattern!r} did not come out {'superlinear' if quadratic else 'linear'}."]
    rows, dynamic, read, unread = census(root)
    if unread:
        return 1, ["Pattern cost: FAILED -- blind: these modules could not be parsed:"] + [f"  {m}" for m in unread]
    again = {}

    def held(pattern, flags, methods, first):
        """(verdict, reading) most of READINGS readings agree on, for a pattern whose first reading goes against
        the ledger."""
        key = (pattern, flags, methods)
        if key not in again:
            again[key] = steady(pattern, flags, methods, first)
        return again[key]

    by_module = {}
    for module, line, pattern, flags, methods, result in rows:
        by_module.setdefault(module, []).append((line, pattern, flags, methods, result))
    for module in sorted(set(by_module) | set(LEDGER)):
        owed = list(LEDGER.get(module, ()))
        found, unowed, linear = [], [], []
        for line, pattern, flags, methods, result in by_module.get(module, []):
            name = digest(pattern, flags)
            if not superlinear(result):
                linear.append((line, name, pattern, flags, methods, result))
            elif name in owed:
                owed.remove(name)
                found.append(name)
            else:
                verdict, reading = held(pattern, flags, methods, result)
                if verdict:
                    found.append(name)
                    unowed.append((line, name, pattern, reading))
        for line, name, pattern, flags, methods, result in linear:
            if name in owed:
                if held(pattern, flags, methods, result)[0]:
                    owed.remove(name)
                    found.append(name)
            elif result is not None and result[0] >= HINT:
                # A reading near the bar, not owed, is read again too: one noisy reading never lets one through.
                verdict, reading = held(pattern, flags, methods, result)
                if verdict:
                    found.append(name)
                    unowed.append((line, name, pattern, reading))
        if unowed:
            lines.append(f"{module}: {len(found)} superlinear pattern(s), {len(LEDGER.get(module, ()))} owed, "
                         f"{len(unowed)} not in the ledger:")
            for line, name, pattern, (ratio, spent, lead, ch) in unowed:
                lines.append(f"  line {line}: {name} x{ratio:.1f} at {spent * 1000:.1f} ms on {lead!r} + {ch!r} x n: "
                             f"{_shown(pattern)}")
        if owed:
            lines.append(f"{module}: the ledger owes {len(LEDGER.get(module, ()))}, the census finds "
                         f"{len(found) - len(unowed)}: lower the entry, remove {', '.join(owed)}")
    if lines:
        return 1, ["Pattern cost: FAILED --"] + lines
    owed = sum(len(names) for names in LEDGER.values())
    return 0, [f"Pattern cost: OK -- {read} site(s) read, {len(rows)} measured, {len(dynamic)} dynamic, {owed} "
               f"superlinear owed in {len(LEDGER)} module(s)"]


def main(argv):
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[2]
    code, lines = run(root)
    for line in lines:
        print(line)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
