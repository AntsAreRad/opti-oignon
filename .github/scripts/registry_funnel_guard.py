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
two summarisers the memory block depends on, and structured output -- and the
ledger below is what remains.

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

# What counts as reaching the client: a request method, or a client object
# from which requests are made.
_CLIENT_CALLS = frozenset({"chat", "generate", "embeddings", "embed"})
_CLIENT_CLASSES = frozenset({"Client", "AsyncClient"})

# Debt that predates the funnel: repo-relative module -> sha256 of its text as
# the debt was enumerated. MAY ONLY SHRINK, and no entry may move.
LEDGER = {
    "opti_oignon/agents/base.py": "adf1a32b6e7705aed2ddfe4d2a10dc0171f4d300cbeef785a2e028ff9ebde900",
    "opti_oignon/agents/dynamic_pipeline.py": "6c6386143ecadff03a91f67e571704458dc558beea4680493973ae9e299f9af6",
    "opti_oignon/api/routes_agent.py": "1728203013819d3aeaab2af3d160a038860c163275f13804022bb99e27fcf2b8",
    "opti_oignon/api/routes_answer_verification.py": "e02e93ed50726d9a1eabfc525f5bac59b7d98f38429d5ef36b845fd292eac0ed",
    "opti_oignon/api/routes_benchmark.py": "5e3900bea555ff38a5229357b294c0ee9a66026f9713a2c445baec74f0cf553a",
    "opti_oignon/api/routes_citation_verification.py": "3097310fe98d52a754df16799deb06713e419b46ef3c32a0dfc121a00ffc7ee5",
    "opti_oignon/api/routes_claim_verification.py": "369fb16980f2daaaa258dcec55b60c511a23ea4a8cdc2fe522b3271120f5d8e9",
    "opti_oignon/api/routes_fine_tune.py": "bb80ad1d5f20c92ba68945a3eb62d9de25bcd04ee6af881304859425eb7794c8",
    "opti_oignon/api/routes_note_actions.py": "9767920c8f68929dab8bbb8e642d426d0461bc5f0f5df3d723e1a11014b694a6",
    "opti_oignon/benchmark_judge.py": "3a4b1eaaba98b3f0f006822e66ed55c472f861b34a66635672a81c3a1a8c1bb7",
    "opti_oignon/benchmark_runner.py": "b0620fdcd46bfe709613af2a2adc297a91a3de251cadee5a071373bde3e0c993",
    "opti_oignon/cascading.py": "7ebd56407919f33f2773bce1825bdb5da0d69da7255d8410ba52e2319e650e2b",
    "opti_oignon/humanizer.py": "5cf648492f279f4cee1f23ff4bf70861bfd8f02c2cc6d07b60ef3ac2ed5da084",
    "opti_oignon/memory/legacy.py": "a7f12990331096707028425d6257480be0cce2acc64148473bb691d639e04dd5",
    "opti_oignon/model_warmup.py": "a7954f8256dae5eac366425373b152cbca911bf329e764ec022d3723c9782581",
    "opti_oignon/pre_cache.py": "146f184dfd5317b53987efaad2f084f72fc2d29f6300bdf1b7148ff849a2d43a",
    "opti_oignon/reasoning.py": "b629a2654f74e060c6a978d75a543d8da8aeaf5eda734c0e6d9feb732c9d5088",
    "opti_oignon/routing/benchmark.py": "33e7f930ffe42d8b53e4deb794775f83898dbee3fc43c62e2e7177cae45b8301",
    "opti_oignon/self_correction.py": "87a6db34185a5f364ba27fb68d8cb757b5cb9c4240f8d796a37cb50afce50deb",
    "opti_oignon/semantic_cache.py": "edd88abf6bd9cc43822e2b6b89802c27fa17c13e3827b7c2934174259c9562e5",
    "opti_oignon/speculative.py": "37db6c66099db9164553a8c7c3f7db12fa8a4f110dbb691fb4f74fa47bf9f951",
}


def digest(text):
    """The seal of a module, taken on the text this guard reads."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def count_sites(text):
    """How many times the text reaches the client library directly.

    Counted on the syntax tree: a call to a request method on the module or
    any alias of it, a call to a request method imported by name from it, or
    the construction of a client object from it. Prose never counts. A text
    that does not parse is counted as zero here -- the syntax tier owns that
    failure and reports it by name.
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
    n = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in aliases
            and func.attr in _CLIENT_CALLS | _CLIENT_CLASSES
        ):
            n += 1
        elif isinstance(func, ast.Name) and func.id in bare:
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
    sites = sum(count_sites(seen[name]) for name in LEDGER if name in seen)
    print(
        f"Registry funnel OK: {len(LEDGER)} module(s) owed, {sites} direct "
        f"site(s) between them, none outside the ledger. The ledger is sealed: "
        f"it may only shrink, and an owed module that changes must migrate."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
