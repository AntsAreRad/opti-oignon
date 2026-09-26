#!/usr/bin/env python3
"""A static census of the package, for the contracts that hold the componion's boundaries.

Nothing here executes the package: every answer is read from the syntax
trees of ``opti_oignon/**/*.py``, so a contract can say what the chat path,
the model's tools or the plugin loader could ever import, without importing
any of them.

* ``closure(roots, overrides=None, eager=False)`` -- every project module
  the ``roots`` reach, the roots included. An import is found by visiting
  statements only (the module, function and class bodies, ``try``, ``if``,
  ``with``, loops and ``match``), never by walking expressions. A relative
  import is resolved against its package; ``from pkg import name`` counts
  ``pkg.name`` when that is a module; every dotted prefix counts, since
  importing a module runs its packages first. ``eager`` follows only the
  statements outside function bodies. A module named by an import but
  without a file is counted, and followed no further. An import by a
  literal name is followed too, as the statement it amounts to:
  ``import_module("a.b")`` and ``__import__("a.b")`` (a relative name
  resolved against a literal package, or against ``__package__``; a
  ``fromlist`` of literal names counted as a ``from`` import), and
  ``find_spec("a.b")``, which imports the package ``a`` it looks in. These
  are found by walking the expressions of the files whose text names one of
  the callees, and only those.
* ``dynamic_imports(modules)`` -- every call to ``__import__``,
  ``import_module``, ``spec_from_file_location`` or ``find_spec`` whose first
  argument is not a literal, and every relative literal whose package cannot
  be read, as ``(module, enclosing function or "<module>", callee)``: what
  the closure cannot follow. ``spec_from_file_location`` loads a file by its
  path: its name, literal or not, is never followed.
* ``strings(files, tokens)`` -- the string constants of ``files`` that hold
  one of ``tokens`` as a word, lowercased and split on ``[a-z]+`` (so
  ``garden_look`` holds ``garden``); with the number of constants read.
* ``importers(package)`` -- the modules outside the package's own directory
  that import it, or any module under it, at any depth.
* ``calls(modules, names)`` -- every attribute call whose attribute is one of
  ``names``, as ``(module, enclosing function or "<module>", name,
  keywords)``.

``overrides`` maps a dotted module name to source text read in place of its
file, or as a module of its own: the witnesses plant a chain or a call
through it without touching the tree. An enclosing function is named with
its classes and outer functions, dotted (``Garden.sow``); a lambda is not a
function of its own. ``contains`` narrows a census to the files whose text
holds one of its words before any is parsed.

What a file imports (its import statements) is kept in one module-level
cache, keyed by path, modification time and size, so every suite in one
process parses a file for its imports once, overrides or not. The trees themselves are not kept: a process that
holds the syntax trees of the whole package pays for them in every later
garbage collection, and a census parses again, after its text filter, the
few files it reads. While a census parses, the cyclic collector is paused
(a tree holds no cycle, and is freed as soon as it is read); it is resumed
as it was found.

Stdlib-only and import-safe. Used by contract suites; never by the package.
"""

import ast
import contextlib
import gc
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE = "opti_oignon"
ROOT = REPO / PACKAGE
DYNAMIC = ("__import__", "import_module", "spec_from_file_location", "find_spec")
# The callees whose literal name the closure follows (``spec_from_file_location`` loads by path, not by name).
LITERAL_CALLEES = ("__import__", "import_module", "find_spec")
_WORD = re.compile(r"[a-z]+")

# (path, (mtime_ns, size)) -> the file's import statements, each with whether it sits in a function body;
# (module, (mtime_ns, size), eager) -> the modules they reach when no override is given.
_STATEMENTS = {}
_IMPORTS = {}
# (path, (mtime_ns, size)) -> a tree a census asked to keep.
_KEPT = {}


@contextlib.contextmanager
def _parsing():
    """The cyclic garbage collector paused while trees are built and read, and resumed as it was found."""
    was = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was:
            gc.enable()


def full(name):
    """``name`` as a dotted project module: ``cli.session`` and ``opti_oignon.cli.session`` alike."""
    return name if name == PACKAGE or name.startswith(PACKAGE + ".") else PACKAGE + "." + name


def modules():
    """Every module of the package, by dotted name: ``{name: path}``.

    ``__pycache__`` and every directory named ``data`` are not walked: the
    package's data directory holds the maintainer's stores, never a module.
    """
    found = {}
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", "data"))
        for file in sorted(files):
            if not file.endswith(".py"):
                continue
            path = Path(folder, file)
            parts = [PACKAGE, *path.relative_to(ROOT).with_suffix("").parts]
            if parts[-1] == "__init__":
                parts = parts[:-1]
            found[".".join(parts)] = path
    return found


def _stamp(path):
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size)


def parse_file(path):
    """The syntax tree of ``path``."""
    path = Path(path)
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _text(name, known, overrides):
    """The source of a module: its override, else its file; ``None`` when it has neither."""
    if overrides and name in overrides:
        return overrides[name]
    path = known.get(name)
    return None if path is None else path.read_text(encoding="utf-8")


def _is_package(name, known):
    path = known.get(name)
    return path is not None and path.name == "__init__.py"


def _prefixes(name):
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts) + 1)]


def _project(name):
    return name == PACKAGE or name.startswith(PACKAGE + ".")


def _base(node, module, is_package):
    """The module a ``from`` import names, resolved against its package; ``None`` when it climbs out."""
    if not node.level:
        return node.module or ""
    parts = module.split(".") if is_package else module.split(".")[:-1]
    if node.level - 1 > len(parts) - 1:
        return None
    parts = parts[:len(parts) - (node.level - 1)]
    return ".".join(parts + ([node.module] if node.module else []))


def resolve(node, module, known, overrides=None):
    """The project modules one import statement reaches, every dotted prefix included."""
    is_package = _is_package(module, known)
    reached = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            if _project(alias.name):
                reached.extend(_prefixes(alias.name))
        return reached
    base = _base(node, module, is_package)
    if not base or not _project(base):
        return reached
    reached.extend(_prefixes(base))
    for alias in node.names:
        child = base + "." + alias.name
        if child in known or (overrides and child in overrides):
            reached.append(child)
    return reached


def statements(body, eager=False, in_function=False):
    """Every import statement of ``body``, visiting statements only; ``eager`` skips function bodies."""
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node, in_function
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not eager:
                yield from statements(node.body, eager, True)
        elif isinstance(node, ast.ClassDef):
            yield from statements(node.body, eager, in_function)
        else:
            for field in ("body", "orelse", "finalbody"):
                inner = getattr(node, field, None)
                if isinstance(inner, list):
                    yield from statements(inner, eager, in_function)
            for handler in getattr(node, "handlers", None) or ():
                yield from statements(handler.body, eager, in_function)
            for case in getattr(node, "cases", None) or ():
                yield from statements(case.body, eager, in_function)


class _Literal(ast.NodeVisitor):
    """The imports by a literal name in a tree, as the ``Import``/``ImportFrom`` statements they amount to.

    ``found`` holds ``(statement, in a function body)``; ``unread`` the calls
    whose relative name has no package this census can read, as
    ``(enclosing function, callee)``.
    """

    def __init__(self, module, is_package):
        self.module, self.is_package = module, is_package
        self.depth, self.stack = 0, []
        self.found, self.unread = [], []

    def _scope(self, node):
        self.depth += 1
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()
        self.depth -= 1

    visit_FunctionDef = _scope
    visit_AsyncFunctionDef = _scope

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def _package(self, node):
        """The package a relative name is resolved against: a literal, ``__package__``, or ``None``."""
        given = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == "package"),
                                                              None)
        if isinstance(given, ast.Constant) and isinstance(given.value, str):
            return given.value
        if isinstance(given, ast.Name) and given.id == "__package__":
            return self.module if self.is_package else self.module.rpartition(".")[0]
        return None

    def visit_Call(self, node):
        callee = _callee(node.func)
        first = node.args[0] if node.args else None
        if callee in ("__import__", "import_module", "find_spec") and isinstance(first, ast.Constant) \
                and isinstance(first.value, str) and first.value:
            name, level = first.value, 0
            if callee == "__import__":
                given = dict(zip(("globals", "locals", "fromlist", "level"), node.args[1:]))
                given.update({k.arg: k.value for k in node.keywords if k.arg})
                level_node = given.get("level")
                if isinstance(level_node, ast.Constant) and isinstance(level_node.value, int):
                    level = level_node.value
                elif level_node is not None:
                    level = None
                fromlist = given.get("fromlist")
                items = [item.value for item in getattr(fromlist, "elts", ())
                         if isinstance(item, ast.Constant) and isinstance(item.value, str)]
                if level is None:
                    self.unread.append((self.where(), callee))
                elif level:
                    self._add(ast.ImportFrom(module=name, names=[ast.alias(name=i) for i in items] or
                                             [ast.alias(name="*")], level=level))
                else:
                    self._add(ast.Import(names=[ast.alias(name=name)]))
                    if items:
                        self._add(ast.ImportFrom(module=name, names=[ast.alias(name=i) for i in items], level=0))
            else:
                if name.startswith("."):
                    package = self._package(node)
                    stripped = name.lstrip(".")
                    if package is None:
                        self.unread.append((self.where(), callee))
                        name = None
                    else:
                        bits = package.rsplit(".", len(name) - len(stripped) - 1)
                        name = bits[0] + ("." + stripped if stripped else "")
                if name is not None:
                    if callee == "find_spec":
                        name = name.rpartition(".")[0]
                    if name:
                        self._add(ast.Import(names=[ast.alias(name=name)]))
        self.generic_visit(node)

    def where(self):
        return ".".join(self.stack) if self.stack else "<module>"

    def _add(self, statement):
        self.found.append((statement, self.depth > 0))


def _literal_imports(tree, name, is_package):
    """The ``_Literal`` census of ``tree``, a module called ``name``."""
    visitor = _Literal(name, is_package)
    visitor.visit(tree)
    return visitor


def _with_literals(tree, text, name, is_package):
    """The import statements of a tree, and the imports by a literal name when its text names a callee."""
    found = tuple(statements(tree.body))
    if any(callee in text for callee in LITERAL_CALLEES):
        found += tuple(_literal_imports(tree, name, is_package).found)
    return found


def _import_statements(name, known, overrides):
    """``((statement, in a function body), ...)`` of a module, or ``None`` when it has no source."""
    is_package = _is_package(name, known)
    if overrides and name in overrides:
        text = overrides[name]
        tree = ast.parse(text, filename="<" + name + ">")
        return _with_literals(tree, text, name, is_package)
    path = known.get(name)
    if path is None:
        return None
    key = (str(path), _stamp(path))
    if key not in _STATEMENTS:
        with _parsing():
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
            _STATEMENTS[key] = _with_literals(tree, text, name, is_package)
    return _STATEMENTS[key]


def imports(name, *, known=None, overrides=None, eager=False):
    """The project modules module ``name`` imports (each import resolved, prefixes included)."""
    known = modules() if known is None else known
    memo = None
    if not overrides and name in known:
        memo = (name, _stamp(known[name]), eager)
        if memo in _IMPORTS:
            return _IMPORTS[memo]
    found = _import_statements(name, known, overrides)
    if found is None:
        return ()
    reached = []
    for node, in_function in found:
        if not (eager and in_function):
            reached.extend(resolve(node, name, known, overrides))
    out = tuple(dict.fromkeys(reached))
    if memo is not None:
        _IMPORTS[memo] = out
    return out


def closure(roots, *, overrides=None, eager=False):
    """Every project module ``roots`` reach, breadth first over the import statements; the roots included."""
    known = modules()
    queue = []
    for root in roots:
        queue.extend(_prefixes(full(root)))
    seen = set()
    with _parsing():
        while queue:
            name = queue.pop(0)
            if name in seen:
                continue
            seen.add(name)
            for reached in imports(name, known=known, overrides=overrides, eager=eager):
                if reached not in seen:
                    queue.append(reached)
    return frozenset(seen)


class _Scoped(ast.NodeVisitor):
    """Visits a tree with the dotted name of the enclosing classes and functions at hand."""

    def __init__(self):
        self.stack = []

    def where(self):
        return ".".join(self.stack) if self.stack else "<module>"

    def _scope(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = _scope
    visit_AsyncFunctionDef = _scope
    visit_ClassDef = _scope


def _callee(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def trees(names, overrides=None, contains=None, keep=False):
    """``[(module, tree)]`` for ``names`` that have a file or an override (and hold a word of ``contains``).

    ``keep`` keeps the trees of files in a cache of their own, for a census
    that reads the same few files again (the being's own modules, say).
    """
    known = modules()
    out = []
    with _parsing():
        for name in dict.fromkeys(full(n) for n in names):
            if keep and not (overrides and name in overrides) and name in known:
                text = known[name].read_text(encoding="utf-8")
                if contains is not None and not any(word in text for word in contains):
                    continue
                key = (str(known[name]), _stamp(known[name]))
                if key not in _KEPT:
                    _KEPT[key] = ast.parse(text, filename=str(known[name]))
                out.append((name, _KEPT[key]))
                continue
            text = _text(name, known, overrides)
            if text is None or (contains is not None and not any(word in text for word in contains)):
                continue
            out.append((name, ast.parse(text, filename="<" + name + ">")))
    return out


def dynamic_imports(names, *, overrides=None):
    """Every non-literal ``__import__``/``import_module``/``spec_from_file_location``/``find_spec`` call, and
    every relative literal one whose package cannot be read: what the closure cannot follow."""
    found = []
    known = modules()
    for module, tree in trees(names, overrides, contains=DYNAMIC):
        visitor = _Scoped()

        def visit_Call(node, visitor=visitor, module=module):
            callee = _callee(node.func)
            if callee in DYNAMIC:
                first = node.args[0] if node.args else None
                if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                    found.append((module, visitor.where(), callee))
            visitor.generic_visit(node)

        visitor.visit_Call = visit_Call
        visitor.visit(tree)
        found.extend((module, where, callee)
                     for where, callee in _literal_imports(tree, module, _is_package(module, known)).unread)
    return sorted(set(found))


def words(text):
    """The lowercase words of ``text``: ``[a-z]+`` after lowercasing."""
    return _WORD.findall(text.lower())


def strings(files, tokens, *, texts=None):
    """``(hits, read)``: ``(file, constant, token)`` for every string constant holding a token, and the count read.

    ``files`` are paths relative to the repository; ``texts`` maps a path to
    the text read in its place.
    """
    tokens = frozenset(tokens)
    hits, read = [], 0
    for file in files:
        text = texts[file] if texts and file in texts else (REPO / file).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                read += 1
                for word in sorted(set(words(node.value)) & tokens):
                    hits.append((file, node.value, word))
    return hits, read


def importers(package=PACKAGE + ".allium", *, overrides=None):
    """The modules outside ``package``'s own directory that import it, or any module under it, at any depth."""
    known = modules()
    names = set(known) | set(overrides or ())
    leaf = package.rsplit(".", 1)[-1]
    found = set()
    with _parsing():
        for name in sorted(names):
            if name == package or name.startswith(package + "."):
                continue
            text = _text(name, known, overrides)
            if leaf not in text:
                continue
            for reached in imports(name, known=known, overrides=overrides):
                if reached == package or reached.startswith(package + "."):
                    found.add(name)
                    break
    return frozenset(found)


def calls(names, methods, *, overrides=None, keep=False):
    """``(module, enclosing function, method, keywords)`` of every attribute call to one of ``methods``."""
    methods = frozenset(methods)
    found = []
    for module, tree in trees(names, overrides, contains=tuple("." + method for method in methods), keep=keep):
        visitor = _Scoped()

        def visit_Call(node, visitor=visitor, module=module):
            if isinstance(node.func, ast.Attribute) and node.func.attr in methods:
                keywords = tuple(sorted(k.arg for k in node.keywords if k.arg is not None))
                found.append((module, visitor.where(), node.func.attr, keywords))
            visitor.generic_visit(node)

        visitor.visit_Call = visit_Call
        visitor.visit(tree)
    return found


def package_modules(prefix):
    """The dotted names of every module at or under ``prefix`` (``opti_oignon.allium``, say)."""
    prefix = full(prefix)
    return sorted(name for name in modules() if name == prefix or name.startswith(prefix + "."))


__all__ = ["REPO", "PACKAGE", "ROOT", "DYNAMIC", "full", "modules", "parse_file", "resolve", "statements",
           "imports", "closure", "trees", "dynamic_imports", "words", "strings", "importers", "calls",
           "package_modules"]
