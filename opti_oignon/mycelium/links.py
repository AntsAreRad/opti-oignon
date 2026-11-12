"""The Mycelium's network, its first kind of edge: a Python file that names, resolved through its imports, what another
file of the same source defines, binds or is.

An edge ``src -> dst`` holds when at least one part of ``src`` makes a reference the cutter recorded for the network
(``chunk.py``: what an import statement brings, a name read that an import bound, a chain of attributes read on one)
and that reference resolves to ``dst``; ``evidence`` counts the parts of ``src`` that do. The other languages keep the
join by name.

A file is importable under its dotted path (``app/a.py`` -> ``app.a``, a package's ``__init__.py`` -> the package),
and, under a leading ``src/`` or ``lib/`` that is no package, under the name without it too (the ``src/`` layout
installed) unless that name would be the standard library's. One name for two files: the package before the module,
then the first by path; a dotted prefix of modules that names no file is a namespace package. A reference resolves
from left to right as the interpreter finds it: its first name a module of the source (else it is outside: nothing);
then each name on the module reached -- defined there (a function, a class, a name assigned at its level): that file;
bound there by an import to a module (a package importing its own submodule): that module, entered; bound there to a
name: that binding's target, followed (three steps at most; past them, that file, which holds the name); a submodule:
the module reached next; an import with a star there: each tried; else nothing (the interpreter
would raise). A target is a module's dotted path, then the names read on it (``from m import n``: the module ``m``,
then ``n``; ``import a.b``: the module ``a.b``): along the path, a submodule comes first (the import system imports
the module), and on the names read, the attribute first (``getattr`` first, as the interpreter does). A
reference that ends on a module is that module's file, but a namespace package (no file) and a package's ``__init__``
that holds nothing (no definition, no assigned name, no binding) are nothing. The file itself is no edge. Names made
at run time (``globals()``, ``setattr``, a module's ``__getattr__``) are not seen.

The edges of a source are computed again, whole, by an index pass that changed, renamed, removed or cut again a file
of it, or that finds them computed by another version of this resolution (``LINKS_VERSION``, kept per source in the
store's meta), or never: exact by construction, in the pass's own transaction -- a pass that fails keeps none of
them. A secret source has none; a source of documents has none. The edges hold two file ids and a count: no word of
the source.

  recompute(store, source, moved=...)  the edges of ``source`` (its id), computed again when due; their counts
  links_of(store, file_id)             (links to, linked from): [(path, evidence)], most first, then by path
  next_files(store, source, opened, n) the walk from the opened files: (linked, same names), each [(path, score)]
"""
import sys
from collections import defaultdict

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

LINKS_VERSION = 1
KINDS = ("imodule", "ifrom", "bound", "dotted")     # the references that make an edge
NAMESPACE = ("binds", "assign")                     # what a module's own namespace holds
MAX_DEFINED = 6         # a name with more definitions in the source joins nothing (the map's users' rule)
CHUNK = 400             # names bound in one query
HOPS = 3                # bindings followed, at most
STDLIB = frozenset(sys.stdlib_module_names)


def module_names(path, packages=frozenset()):
    """The names a Python file is importable under: its dotted path (``src/lib2/m.py`` -> ``src.lib2.m``; a package's
    ``__init__.py`` -> the package), and, under a leading ``src/`` or ``lib/`` that is no package (``packages``: the
    top folders that hold an ``__init__.py``), the name without that folder too, unless its first name is the
    standard library's (``lib/json.py`` -> only ``lib.json``)."""
    parts = path[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    names = [".".join(parts)] if parts else []
    if len(parts) > 1 and parts[0] in ("src", "lib") and parts[0] not in packages and parts[1] not in STDLIB:
        names.append(".".join(parts[1:]))
    return names


def _target(name, kind, module):
    """The target of one recorded reference, ``path:names`` -- a module by its dotted path, then the names read on it
    (none for a module itself): a chain's rest read on its head's binding, else the target the cutter recorded."""
    if kind == "dotted":
        return _on(module, name.partition(".")[2].split("."))
    return module


def _on(target, names):
    """``target`` (a module path, or ``path:names``) with more ``names`` read on it."""
    names = [n for n in names if n]
    if not names:
        return target
    return f"{target}.{'.'.join(names)}" if ":" in target else f"{target}:{'.'.join(names)}"


class _Resolver:
    """The modules of one source and each one's namespace, and the resolution of a dotted target."""

    def __init__(self, files, defined, binds, stars):
        packages = {path.split("/", 1)[0] for _f, path in files if path.count("/") == 1
                    and path.endswith("/__init__.py")}
        self.inits = {f for f, path in files if path.endswith("__init__.py")}
        held = defaultdict(list)
        for file_id, path in files:
            for name in module_names(path, packages):
                held[name].append((file_id not in self.inits, path, file_id))
        self.modules = {name: min(found)[2] for name, found in held.items()}   # the package first, then by path
        self.spaces = {".".join(name.split(".")[:k]) for name in self.modules for k in range(1, name.count(".") + 1)}
        self.spaces -= set(self.modules)
        self.defined, self.binds, self.stars = defined, binds, stars

    def _holds(self, file_id):
        return bool(self.defined.get(file_id) or self.binds.get(file_id) or self.stars.get(file_id))

    def _module(self, name):
        """(file, True) for a module of the source, (None, True) for a namespace package, (None, False) otherwise."""
        if name in self.modules:
            return self.modules[name], True
        return None, name in self.spaces

    def resolve(self, target, hops=HOPS):
        """The file ``target`` (``path:names``) resolves to in the source, or None (the rule in this module's
        docstring)."""
        path, _, names = target.partition(":")
        parts = path.split(".") + (names.split(".") if names else [])
        modular = path.count(".") + 1           # the first names are a module's path: its submodules first
        here = parts[0]
        current, found = self._module(here)
        if not found:
            return None
        for i in range(1, len(parts)):
            name, rest = parts[i], parts[i + 1:]
            nxt, sub = self._module(f"{here}.{name}")
            if not (i < modular and sub):
                if name in self.defined.get(current, ()):     # a namespace package (no file) defines nothing
                    return current
                bound = self.binds.get(current, {}).get(name)
                entered = bound.replace(":", ".") if bound is not None else None
                if bound is not None and (entered in self.modules or entered in self.spaces):
                    # The binding names a module (a package importing its own submodule, `from . import sub`):
                    # that module is entered, as the attribute would give it, and the walk goes on from it.
                    current, here = self.modules.get(entered), entered
                    continue
                if bound is not None:
                    return current if hops <= 0 else self.resolve(_on(bound, rest), hops=hops - 1)
            if sub:
                current, here = nxt, f"{here}.{name}"
                continue
            for star in self.stars.get(current, ()) if hops > 0 else ():
                hit = self.resolve(_on(star, parts[i:]), hops=hops - 1)
                if hit is not None:
                    return hit
            return None
        if current in self.inits and not self._holds(current):
            return None
        return current      # None for a namespace package: no file


def _edges(db, source):
    """{(src, dst): evidence} of ``source`` from its recorded references, every row read once."""
    files = db.execute("SELECT id, path FROM file WHERE source = ? AND lang = 'python' ORDER BY path",
                       (source,)).fetchall()
    defined, binds, stars = defaultdict(set), defaultdict(dict), defaultdict(list)
    for file_id, name in db.execute(
            "SELECT p.file, y.name FROM symbol y JOIN part p ON p.id = y.part JOIN file f ON f.id = p.file "
            "WHERE f.source = ? AND f.lang = 'python' AND y.kind IN ('function', 'class')", (source,)):
        defined[file_id].add(name)
    rows = db.execute(
        "SELECT p.file, p.id, r.name, r.kind, r.module FROM ref r JOIN part p ON p.id = r.part "
        "JOIN file f ON f.id = p.file WHERE f.source = ? AND f.lang = 'python' AND r.kind IN (?, ?, ?, ?, ?, ?) "
        "ORDER BY p.file, p.id, r.rowid", (source, *KINDS, *NAMESPACE)).fetchall()
    for file_id, _part, name, kind, module in rows:
        if kind == "assign":
            defined[file_id].add(name[1:])
        elif kind == "binds" and name == "=*":
            stars[file_id].append(module)
        elif kind == "binds":
            binds[file_id].setdefault(name[1:], module)
    resolver = _Resolver(files, defined, binds, stars)
    parts = defaultdict(set)
    for file_id, part_id, name, kind, module in rows:
        if kind not in KINDS:
            continue
        dst = resolver.resolve(_target(name, kind, module))
        if dst is not None and dst != file_id:
            parts[(file_id, dst)].add(part_id)
    return {pair: len(held) for pair, held in parts.items()}


def recompute(store, source, *, moved):
    """The edges of ``source`` (its id) computed again, whole, in the caller's transaction, when ``moved`` (its pass
    changed, renamed, removed or cut again a file) or when they were computed by another version of this resolution,
    or never: the rows of edges that went deleted, the new ones written, the version kept. A secret source has none
    (its rows deleted). Returns {"links": edges held, "dropped": edges deleted}, or None when nothing was due -- and
    then nothing is written."""
    if not moved and store.meta(f"links:{source}") == str(LINKS_VERSION):
        return None
    db = store.db
    secret = db.execute("SELECT sensitivity = 'secret' FROM source WHERE id = ?", (source,)).fetchone()[0]
    held = {(a, b): n for a, b, n in db.execute(
        "SELECT l.src, l.dst, l.evidence FROM link l JOIN file f ON f.id = l.src WHERE f.source = ?", (source,))}
    wanted = {} if secret else _edges(db, source)
    gone = [pair for pair in held if pair not in wanted]
    db.executemany("DELETE FROM link WHERE src = ? AND dst = ?", gone)
    changed = [(n, a, b) for (a, b), n in wanted.items() if (a, b) in held and held[(a, b)] != n]
    db.executemany("UPDATE link SET evidence = ? WHERE src = ? AND dst = ?", changed)
    db.executemany("INSERT INTO link(src, dst, evidence) VALUES (?, ?, ?)",
                   [(a, b, n) for (a, b), n in wanted.items() if (a, b) not in held])
    if secret:
        db.execute("DELETE FROM meta WHERE key = ?", (f"links:{source}",))     # computed again once it is not
    else:
        store.meta(f"links:{source}", LINKS_VERSION)
    return {"links": len(wanted), "dropped": len(gone)}


def _has_links(db):
    return db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'link'").fetchone() is not None


def links_of(store, file_id):
    """(links to, linked from) of a file: [(path, evidence)], most parts first, then by path; none from a store of
    schema 3 read as it is (no link table)."""
    db = store.db
    if not _has_links(db):
        return [], []
    to = db.execute("SELECT f.path, l.evidence FROM link l JOIN file f ON f.id = l.dst WHERE l.src = ? "
                    "ORDER BY l.evidence DESC, f.path", (file_id,)).fetchall()
    back = db.execute("SELECT f.path, l.evidence FROM link l JOIN file f ON f.id = l.src WHERE l.dst = ? "
                      "ORDER BY l.evidence DESC, f.path", (file_id,)).fetchall()
    return [tuple(r) for r in to], [tuple(r) for r in back]


def next_files(store, source, opened, limit):
    """The walk from the files ``opened`` (ids of ``source``): (linked, same names), each [(path, score)], ``limit``
    at most, most first then by path, the opened left out. Linked: each edge between an opened file and another
    counts once each way. Same names, the map's users' rule: a name an opened file references and another file
    defines (a definition of any kind but a module's), or the reverse, counts once per (opened, other) pair and way;
    a name with more than ``MAX_DEFINED`` definitions in the source, of any kind, counts for nothing; a file the
    linked list shows is not shown again (one it leaves past ``limit`` may be)."""
    db = store.db
    opened = sorted(set(opened))
    if not opened:
        return [], []
    held = set(opened)
    marks = ",".join("?" * len(opened))
    linked = defaultdict(int)
    if _has_links(db):
        for (other,) in db.execute(f"SELECT dst FROM link WHERE src IN ({marks})", opened):
            linked[other] += 1
        for (other,) in db.execute(f"SELECT src FROM link WHERE dst IN ({marks})", opened):
            linked[other] += 1
    for file_id in held:
        linked.pop(file_id, None)
    path_of = dict(db.execute("SELECT id, path FROM file WHERE source = ?", (source,)))
    shown = _ranked(linked, path_of, limit)     # ids, most first then by path
    # The names that join an opened file to another -- those it references, those it defines --, kept when the source
    # holds at most MAX_DEFINED definitions of them.
    wanted = {r[0] for r in db.execute(f"SELECT DISTINCT r.name FROM ref r JOIN part p ON p.id = r.part "
                                       f"WHERE p.file IN ({marks})", opened)}
    wanted |= {r[0] for r in db.execute(f"SELECT DISTINCT y.name FROM symbol y JOIN part p ON p.id = y.part "
                                        f"WHERE p.file IN ({marks}) AND y.kind != 'module'", opened)}
    names = sorted(wanted)
    kept, definers, users = set(), defaultdict(set), defaultdict(set)
    for at in range(0, len(names), CHUNK):
        chunk = names[at:at + CHUNK]
        some = ",".join("?" * len(chunk))
        for name, count in db.execute(
                f"SELECT y.name, COUNT(*) FROM symbol y JOIN part p ON p.id = y.part JOIN file f ON f.id = p.file "
                f"WHERE f.source = ? AND y.name IN ({some}) GROUP BY y.name", (source, *chunk)):
            if count <= MAX_DEFINED:
                kept.add(name)
    kept = sorted(kept)
    for at in range(0, len(kept), CHUNK):
        chunk = kept[at:at + CHUNK]
        some = ",".join("?" * len(chunk))
        for name, file_id in db.execute(
                f"SELECT DISTINCT y.name, p.file FROM symbol y JOIN part p ON p.id = y.part JOIN file f "
                f"ON f.id = p.file WHERE f.source = ? AND y.kind != 'module' AND y.name IN ({some})", (source, *chunk)):
            definers[name].add(file_id)
        for name, file_id in db.execute(
                f"SELECT DISTINCT r.name, p.file FROM ref r JOIN part p ON p.id = r.part JOIN file f "
                f"ON f.id = p.file WHERE f.source = ? AND r.name IN ({some})", (source, *chunk)):
            users[name].add(file_id)
    pairs = set()
    for name, defined_in in definers.items():
        for user in users.get(name, ()):
            for definer in defined_in:
                if user != definer and (user in held or definer in held):
                    pairs.add((user, definer))
    named = defaultdict(int)
    for user, definer in pairs:
        if user in held and definer not in held:
            named[definer] += 1
        if definer in held and user not in held:
            named[user] += 1
    for file_id in shown:
        named.pop(file_id, None)
    return ([(path_of[f], linked[f]) for f in shown],
            [(path_of[f], named[f]) for f in _ranked(named, path_of, limit)])


def _ranked(scores, path_of, limit):
    return sorted(scores, key=lambda f: (-scores[f], path_of[f]))[:limit]
