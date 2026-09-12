#!/usr/bin/env python3
"""Registry-funnel guard: every inference request goes through the registry.

BackendRegistry is where the guarantees live. A request that passes through it
is admitted by the resource governor, may carry a schema or a tool list as an
engine option, is served against a probed VRAM capacity, and comes back with a
figure that knows where it came from. A request that reaches the client
library directly -- ``ollama.chat``, an alias of it, a name imported from it,
or a client constructed from it -- gets none of that, and there is no log line
to say so. When this guard was written, twenty-nine modules did exactly that,
at fifty-one sites; four were paid in the same block -- the funnel itself, the
two summarisers the memory block depends on, and structured output -- nine
more in the next, and the last eleven in the third convergence block. The
fourth widened what "reaching the client" means -- a receiver the client
module was bound to, a request method handed on uncalled, and the catalogue
reads ``list`` and ``show`` -- and found twenty-two more sites in sixteen
modules, all paid in that block. The ledger below is empty, and it stays
empty: a direct site anywhere outside the funnel is a violation by name.

RATCHET, in the shape of the isolation-seal guard and for the same reason: a
ratchet that only counts is a ratchet on the count. Every owed module carries
the digest of its text as the debt was enumerated. An owed module that changes
while still calling the client directly no longer matches its seal and becomes
a violation: touch it, and you migrate it. The debt is frozen as it was found,
it can be paid, and it cannot grow -- not in modules, and not in lines.

The census is taken on the syntax tree, never on the text. A docstring that
mentions the client is not a request, and a guard that charged for prose would
be green or red for reasons unrelated to what leaves the process.

One module is exempt by name: ``opti_oignon/inference_backend.py`` is where the
registry's own Ollama backend talks to the client. That is the funnel; it is
not a bypass of the funnel.

Three questions, three answers, with disjoint domains so no one can cover for
another:

  * ``find_violations``           -- a direct caller nobody owes for.
  * ``find_broken_seals``         -- an owed module whose bytes moved.
  * ``find_stale_ledger_entries`` -- an owed module that migrated, or vanished.

The helpers are pure and import-safe; ``main`` scans the package and exits
non-zero on any finding. Usage: ``registry_funnel_guard.py [REPO_ROOT]``.
"""

import ast
import hashlib
import sys
from pathlib import Path

_PACKAGE_DIR = "opti_oignon"
_FUNNEL = "opti_oignon/inference_backend.py"

# What counts as reaching the client: a request method, a read of the
# engine's loaded set, a read of the model catalogue, or a client object
# from which requests are made. ``ps`` joined the set when the loaded set
# became a head on the backend contract; ``list`` and ``show`` joined it in
# the fourth convergence block, when the catalogue went through
# ``list_models`` and ``model_info``: a module that reads any of them from
# the client bypasses the head that answers it. Model management --
# ``pull``, ``delete``, ``copy``, ``create``, ``push`` -- has no head on the
# contract and is not counted: whether it belongs in the funnel is a
# decision the guard does not take on its own.
_CLIENT_CALLS = frozenset({"chat", "generate", "embeddings", "embed", "ps", "list", "show"})
_CLIENT_CLASSES = frozenset({"Client", "AsyncClient"})

# Debt that predates the funnel: repo-relative module -> sha256 of its text as
# the debt was enumerated. MAY ONLY SHRINK, and no entry may move. It is
# empty since the third convergence block paid the last eleven modules: a
# direct site anywhere outside the funnel is now a violation by name, and
# nothing may be added here to make one tolerable.
LEDGER = {
}


def digest(text):
    """The seal of a module, taken on the text this guard reads."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _names_in(node):
    """Every bare name read anywhere inside an expression."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def count_sites(text):
    """How many times the text reaches the client library directly.

    Counted on the syntax tree: a request method or a client class reached
    on the module, on any alias of it, on a name or an attribute the module
    was bound to, or imported by name from it. A reference counts whether or
    not it is called: a request method handed on as a callable is a route to
    the client. Prose never counts. A text that does not parse is counted as
    zero here -- the syntax tier owns that failure and reports it by name.

    Binding is followed one assignment at a time until nothing new binds:
    ``self._c = injected or _ollama`` makes ``_c`` a client attribute, and
    ``c = ollama.Client(host)`` makes ``c`` a client name. The receiver is
    what the fourth convergence block found the census blind to, with a
    ``chat`` request behind it.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0
    aliases = set()
    bare = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ollama":
                    aliases.add(alias.asname or "ollama")
        elif isinstance(node, ast.ImportFrom) and node.module == "ollama":
            for alias in node.names:
                if alias.name in _CLIENT_CALLS | _CLIENT_CLASSES:
                    bare.add(alias.asname or alias.name)
    if not aliases and not bare:
        return 0
    attrs = set()
    grown = True
    while grown:
        grown = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            tainted = _names_in(value) & (aliases | bare)
            if not tainted:
                tainted = {
                    n.attr for n in ast.walk(value)
                    if isinstance(n, ast.Attribute) and n.attr in attrs
                }
            if not tainted:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in aliases:
                    aliases.add(target.id)
                    grown = True
                elif isinstance(target, ast.Attribute) and target.attr not in attrs:
                    attrs.add(target.attr)
                    grown = True
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _CLIENT_CALLS | _CLIENT_CLASSES:
            value = node.value
            if isinstance(value, ast.Name) and value.id in aliases:
                n += 1
            elif isinstance(value, ast.Attribute) and value.attr in attrs:
                n += 1
        elif isinstance(node, ast.Name) and node.id in bare and isinstance(node.ctx, ast.Load):
            n += 1
    return n


def calls_directly(name, text):
    """True when the module reaches the client and is not the funnel itself."""
    return name != _FUNNEL and count_sites(text) > 0


def find_violations(files):
    """Modules that call the client directly and that the ledger does not owe for.

    ``files`` is an iterable of (repo-relative name, text) pairs. An owed name
    is passed over HERE and answered for by the seal below; the domains are
    disjoint by construction.
    """
    return sorted(
        name for name, text in files
        if calls_directly(name, text) and name not in LEDGER
    )


def find_broken_seals(files):
    """Owed modules whose bytes no longer match their seal, still calling directly.

    The ratchet's tooth. A module that migrated is NOT broken -- migrating is
    what the seal asks for; it becomes stale instead and comes off the ledger.
    """
    seen = dict(files)
    return sorted(
        name for name, sealed in LEDGER.items()
        if name in seen
        and calls_directly(name, seen[name])
        and digest(seen[name]) != sealed
    )


def find_stale_ledger_entries(files):
    """Ledger names that no longer call the client directly, or that vanished.

    An entry that has been paid must come OFF the list, or the debt count stops
    meaning anything and a later regression could hide behind it.
    """
    seen = dict(files)
    return sorted(
        name for name in LEDGER
        if name not in seen or not calls_directly(name, seen[name])
    )


def _estate(root):
    package = Path(root) / _PACKAGE_DIR
    return [
        (p.relative_to(root).as_posix(), p.read_text(encoding="utf-8", errors="ignore"))
        for p in sorted(package.rglob("*.py"))
    ]


def main(argv):
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[2]
    files = _estate(root)
    if not files:
        # A guard that scanned nothing has proven nothing. The zero it would
        # otherwise print is the silent kind this repository treats as a
        # defect, so an empty estate is a refusal, not a pass.
        print(f"Registry funnel: no Python module found under {root / _PACKAGE_DIR}; nothing was scanned.")
        return 1
    violations = find_violations(files)
    broken = find_broken_seals(files)
    stale = find_stale_ledger_entries(files)

    if violations:
        print("Registry-funnel violations -- these modules reach the client")
        print("library directly instead of going through BackendRegistry:")
        for name in violations:
            print(f"  {name}")
        print()
        print("A request that bypasses the registry is not admitted by the")
        print("governor, cannot carry a schema or a tool list, and comes back")
        print("with no provenance. Route it through the registry.")
    if broken:
        print("Broken seals -- the ledger owes for these modules and their bytes")
        print("have moved while they still call the client directly:")
        for name in broken:
            print(f"  {name}")
        print()
        print("Touch an owed module and you migrate it. The debt may be")
        print("carried; it may not be added to.")
    if stale:
        print("Stale ledger entries -- these have been paid or have vanished and")
        print("must be removed from LEDGER so the debt count stays honest:")
        for name in stale:
            print(f"  {name}")

    if violations or broken or stale:
        return 1

    seen = dict(files)
    if not LEDGER:
        # The green with its denominator: how many modules were read to find
        # no direct site, so that a scan of the wrong tree cannot pass as a
        # paid debt.
        print(
            f"Registry funnel OK: 0 module(s) owed, {len(files)} module(s) "
            f"scanned and none reaches the client outside the funnel. The "
            f"ledger is empty and sealed: it may only shrink, so a new direct "
            f"site is a violation by name."
        )
        return 0
    sites = sum(count_sites(seen[name]) for name in LEDGER if name in seen)
    print(
        f"Registry funnel OK: {len(LEDGER)} module(s) owed, {sites} direct "
        f"site(s) between them, none outside the ledger. The ledger is sealed: "
        f"it may only shrink, and an owed module that changes must migrate."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
