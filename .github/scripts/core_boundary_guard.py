#!/usr/bin/env python3
"""Core-boundary guard: the resident core is a named list, and what it pulls
from outside itself is a debt that may only shrink.

The static census drew the line. Without the package facade, the inference
registry, the memory core and the tool dispatch pull nothing outside the
core at module scope; the chat hub pulls dozens. This guard names the core
module by module, records for each core module the outside modules it
imports at module scope today, and holds three rules:

  * a core module that imports a new outside module at module scope is
    refused -- an import inside a function is not a leak, it is a cost paid
    where it happens;
  * a recorded leak the module no longer makes is reported stale and must
    come off the ledger, so the debt count stays honest;
  * no core module reaches the inference client directly -- the registry
    module excepted, since it is the funnel every request goes through.

The census script is the counter, imported by path, so the guard and the
instrument cannot disagree on what an eager import is. Run with no
arguments from the repository root; exit 1 on any refusal.
"""

import importlib.util
import sys
from pathlib import Path

_CENSUS = Path(__file__).resolve().parents[2] / "scripts" / "core_census.py"
PACKAGE = "opti_oignon"
FUNNEL = "opti_oignon.inference_backend"

# The resident core: what must be loaded for one chat turn to be served with
# its guarantees -- admission, provenance, schema, memory, tools, auth -- and
# the primitives those modules pull at module scope.
CORE = frozenset({
    "opti_oignon.__version__",
    "opti_oignon.auth",
    "opti_oignon.config",
    "opti_oignon.context_manager",
    "opti_oignon.conversation",
    "opti_oignon.db_encryption",
    "opti_oignon.db_utils",
    "opti_oignon.emergency_stop",
    "opti_oignon.encryption",
    "opti_oignon.executor",
    "opti_oignon.inference_backend",
    "opti_oignon.memory.composer",
    "opti_oignon.memory.core_store",
    "opti_oignon.memory.drift",
    "opti_oignon.memory.ledger_store",
    "opti_oignon.memory.librarian",
    "opti_oignon.memory.peels",
    "opti_oignon.memory.probes",
    "opti_oignon.memory.receipts",
    "opti_oignon.resource_governor",
    "opti_oignon.response_hygiene",
    "opti_oignon.secure_bytes",
    "opti_oignon.security_mode",
    "opti_oignon.structured_output",
    "opti_oignon.tool_calling",
    "opti_oignon.tool_executor",
    "opti_oignon.tool_registry",
})

# Debt that predates the guard: core module -> the outside modules it imports
# at module scope as the debt was enumerated. MAY ONLY SHRINK. A paid leak
# comes off; a new leak is refused before it is written here.
LEDGER = {
    # The chat hub: twenty-nine outside modules at module scope, most of them
    # guarded imports that decide a feature flag. The hub's diet is a block of
    # its own; the debt is written down here so it can only go down.
    "opti_oignon.executor": frozenset({
        "opti_oignon.agent.untrusted_context",
        "opti_oignon.analyzer",
        "opti_oignon.cascading",
        "opti_oignon.context_dedup",
        "opti_oignon.context_ledger",
        "opti_oignon.context_optimizer",
        "opti_oignon.context_summary",
        "opti_oignon.context_summary_tiers",
        "opti_oignon.context_window",
        "opti_oignon.conversation_compressor",
        "opti_oignon.memory",
        "opti_oignon.memory.auto_capture",
        "opti_oignon.memory.retrieval",
        "opti_oignon.model_warmup",
        "opti_oignon.network_manager",
        "opti_oignon.performance_monitor",
        "opti_oignon.project_context",
        "opti_oignon.project_triggers",
        "opti_oignon.projects",
        "opti_oignon.prompt_optimization",
        "opti_oignon.response_cache",
        "opti_oignon.router",
        "opti_oignon.semantic_cache",
        "opti_oignon.slot_affinity",
        "opti_oignon.speculative",
        "opti_oignon.sync_queue",
        "opti_oignon.token_counter",
        "opti_oignon.verification",
        "opti_oignon.vision_pipeline",
    }),
}


def _census():
    spec = importlib.util.spec_from_file_location("core_census", str(_CENSUS))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.census


def evaluate(root, core, ledger, funnel):
    """The report for the package at ``root`` against ``core`` and ``ledger``."""
    outside = sorted(set(ledger) - set(core))
    if outside:
        raise ValueError(f"the ledger names modules outside the core: {outside}")
    report = _census()(root)
    package = report["package"]
    modules = report["modules"]
    leaks, new_leaks, stale, sites = {}, {}, {}, {}
    for name in sorted(core):
        entry = modules.get(name)
        if entry is None:
            continue
        found = sorted(
            imp for imp in entry["imports_eager"]
            if imp not in core and imp != package
        )
        leaks[name] = found
        owed = set(ledger.get(name, ()))
        fresh = sorted(set(found) - owed)
        paid = sorted(owed - set(found))
        if fresh:
            new_leaks[name] = fresh
        if paid:
            stale[name] = paid
        if name != funnel and entry["direct_client_sites"]:
            sites[name] = int(entry["direct_client_sites"])
    for name in sorted(set(ledger) - set(modules)):
        stale[name] = sorted(ledger[name])
    return {
        "leaks": leaks,
        "new_leaks": new_leaks,
        "stale": stale,
        "client_sites": sites,
        "debt": sum(len(v) for v in ledger.values()),
        "core_present": sorted(n for n in core if n in modules),
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(argv[0]) if argv else Path(__file__).resolve().parents[2] / PACKAGE
    report = evaluate(root, CORE, LEDGER, FUNNEL)
    if report["new_leaks"]:
        print("Core-boundary violations -- these core modules import a module outside")
        print("the core at module scope that the ledger does not carry:")
        for name, leaks in report["new_leaks"].items():
            for leak in leaks:
                print(f"  {name} -> {leak}")
        print()
        print("A leak at module scope makes the whole core pay for a pack's")
        print("dependency at import. Import it where it is used, or move the")
        print("module into the core with its reason.")
    if report["stale"]:
        print("Stale ledger entries -- these leaks have been paid and must be")
        print("removed from LEDGER so the debt count stays honest:")
        for name, leaks in report["stale"].items():
            for leak in leaks:
                print(f"  {name} -> {leak}")
    if report["client_sites"]:
        print("Direct client sites inside the core -- only the funnel may reach")
        print("the client library:")
        for name, count in report["client_sites"].items():
            print(f"  {name}: {count} site(s)")
    if report["new_leaks"] or report["stale"] or report["client_sites"]:
        return 1
    print(
        f"Core boundary OK: {len(report['core_present'])} core module(s), "
        f"{report['debt']} recorded leak(s) between them, none new, none stale, "
        f"no client site outside the funnel. The ledger is sealed: it may only shrink."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
