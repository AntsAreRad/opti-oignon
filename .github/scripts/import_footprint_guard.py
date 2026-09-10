#!/usr/bin/env python3
"""Import footprint guard: a capability costs nothing until it is called.

`.claude/rules/python-modules.md` already states the rule -- "importing the
package creates no database, starts no thread, and touches no network". The
thread clause holds. The database clause does not: importing the package
opens twenty databases through singletons built at module scope, with
encryption not enforced on any of them. In the strict mode the first would
raise at the encryption check and take the import down with it.

A guard that simply asserted the written rule would therefore be red on the
tree it ships with, and a guard red on arrival gets silenced by weakening it.
So this one carries the debt that predates it, exactly as the isolation guard
carries its own: the LEDGER below records what the import does TODAY, and

    THE LEDGER MAY ONLY SHRINK.

Nothing new may appear while the recorded debt is paid down. A database, a
file or a heavy import that is not already on the ledger is refused. An entry
the import no longer reaches is reported stale, so the count cannot drift
away from the debt it claims to measure.

WHAT IS PINNED AND WHAT IS ONLY RECORDED. Decided by measuring the spread,
not by treating the figures as symmetric:

  * The module count is byte-stable -- 2814 across five consecutive runs. It
    carries a ceiling.
  * Wall time is not. Two independent sets of readings on an idle machine,
    2357-2885 ms and 2832-3247 ms, do not overlap in the middle, a spread of
    roughly 15 per cent within each set. A ceiling on it would either forbid
    nothing or flicker with the machine's speed, which is the defect this
    repository named and removed in the browser specs. It is RECORDED.
  * Resident memory is machine-dependent for the same reason at a smaller
    amplitude. Recorded.

The observation is taken in a SUBPROCESS, from an empty directory, through
an audit hook. Never by walking the tree: `data/` holds real personal content
and is not this guard's to read. The hook reports the basename of each
database the import opens, and nothing else about it.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent

# Debt that predates this guard, measured by audit hook on the tree at the
# time it was written. MAY ONLY SHRINK. Removing an entry is how the debt is
# paid; adding one is how the guard is defeated.
LEDGER = {
    # Eight remain, and none yields to the pattern that paid the other
    # twelve. Four live under the data directory, which this guard may not
    # read. One writes outside the tree. One is opened twice by two different
    # owners. One has no connection helper to ensure anything in. And one --
    # branches.db -- was deferred and then put back: its module-level
    # availability flag is false only when construction RAISES, and
    # construction raised because it opened this database. Deferring it lost
    # a fail-secure property rather than merely moving a cost.
    "databases": frozenset({
        "audit_chain.db",
        "branches.db",
        "auth.db",
        "fingerprint.db",
        "humanizer_feedback.db",
        "learned_router.db",
        "plugins.db",
        "sandbox_audit.db",
    }),
    # PAID. The preference store's path was configured as a bare filename
    # and resolved against the caller's directory; it is now anchored on the
    # package data directory. The ledger may only shrink, and this is what
    # shrinking looks like: an entry removed because the code earned it.
    "files": frozenset(),
    # sklearn, and pandas and scipy behind it, are PAID: the classifier's
    # dependency is located rather than imported, and the concrete names are
    # imported in the three methods that use them. That took 1181 modules and
    # 104 MiB off the import.
    "heavy": frozenset({
        "chromadb", "fastapi", "llama_cpp", "numpy", "pydantic",
    }),
}

# Above the 1633 measured -- stable to the digit across three readings -- and
# close enough that a new subtree cannot hide under it. A ceiling set at the
# measurement itself would fail on the first honest import. It was 2900 when
# the guard shipped; the classifier's deferral took 1181 modules off, and a
# ceiling left at the old figure would have stopped forbidding anything.
MODULE_CEILING = 1700

# The import must not start a thread. This one the tree already satisfies.
THREAD_CEILING = 1

_PROBE = '''
import json, os, pathlib, sys, threading

opened = []
before = {p.name for p in pathlib.Path(".").iterdir()}


def hook(event, args):
    if event == "sqlite3.connect" and args:
        opened.append(os.path.basename(str(args[0])))


sys.addaudithook(hook)

import time
started = time.perf_counter()
import opti_oignon  # noqa: F401
elapsed = (time.perf_counter() - started) * 1000

import resource
HEAVY = ("sklearn", "pandas", "scipy", "chromadb", "numpy", "torch",
         "transformers", "sentence_transformers", "onnxruntime", "llama_cpp",
         "fastapi", "pydantic", "uvicorn")
sys.stdout.write("<<FOOTPRINT>>" + json.dumps({
    "databases": sorted(set(opened)),
    "files": sorted({p.name for p in pathlib.Path(".").iterdir()} - before),
    "heavy": sorted({n.split(".")[0] for n in sys.modules
                     if n.split(".")[0] in HEAVY}),
    "threads": threading.active_count(),
    "modules": len(sys.modules),
    "import_ms": round(elapsed, 1),
    "rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
}))
'''


def observe(root=None):
    """Import the package in a subprocess and report what it cost.

    Returns None when the import could not be run at all. A caller must treat
    that as a refusal: an observation that did not happen is not an
    observation of a tree that owes nothing.
    """
    root = Path(root) if root else _ROOT
    with tempfile.TemporaryDirectory() as empty:
        # The environment is inherited, not replaced. Replacing it drops
        # the user site directory, and the package then fails to import for
        # a reason that has nothing to do with its footprint -- which would
        # be measured as "could not run" and refused, wrongly.
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", _PROBE],
            cwd=empty, capture_output=True, text=True, env=env,
        )
    marker = "<<FOOTPRINT>>"
    if marker not in result.stdout:
        return None
    return json.loads(result.stdout.split(marker, 1)[1])


def stale_entries(seen):
    """Ledger entries the import no longer reaches; they must come off."""
    stale = []
    for key in ("databases", "files", "heavy"):
        for name in sorted(LEDGER[key] - set(seen.get(key) or ())):
            stale.append(f"{key}: {name} is on the ledger and no longer "
                         "reached; remove it so the debt count stays honest")
    return stale


def verdict(seen):
    """Reasons to refuse this observation; empty means nothing new."""
    if seen is None:
        return ["the import could not be run, so nothing was measured; an "
                "observation that did not happen is not a clean tree"]

    reasons = []
    modules = seen.get("modules") or 0
    if modules <= 0:
        reasons.append(
            "the probe measured nothing: no module was reported loaded, and "
            "nothing reads exactly like a tree that owes nothing"
        )
        return reasons

    for key, what in (("databases", "database opened at import"),
                      ("files", "file created in the working directory"),
                      ("heavy", "heavy module loaded at import")):
        for name in sorted(set(seen.get(key) or ()) - LEDGER[key]):
            reasons.append(
                f"{what}: {name} is not on the ledger. The ledger may only "
                "shrink; a capability costs nothing until it is called."
            )

    threads = seen.get("threads") or 0
    if threads > THREAD_CEILING:
        reasons.append(
            f"{threads} threads at import, ceiling {THREAD_CEILING}: an "
            "import that starts a thread has done work nobody asked for"
        )

    if modules > MODULE_CEILING:
        reasons.append(
            f"{modules} modules loaded at import, ceiling {MODULE_CEILING}"
        )
    return reasons


def main(argv=None):
    """Observe the import and refuse anything the ledger does not carry."""
    seen = observe()
    reasons = verdict(seen)
    stale = stale_entries(seen) if seen else []

    if reasons:
        print("import footprint guard: FAILED")
        for reason in reasons:
            print(f"  {reason}")
        return 1

    print(
        "import footprint guard: "
        f"{len(seen['databases'])} database(s), {len(seen['files'])} file(s) "
        f"and {len(seen['heavy'])} heavy module(s) at import, all on the "
        f"ledger; {seen['modules']} modules under the {MODULE_CEILING} "
        f"ceiling; {seen['threads']} thread"
    )
    print(
        f"  recorded, not enforced: {seen['import_ms']} ms, "
        f"{seen['rss_mib']} MiB -- both move with the machine, so neither "
        "carries a ceiling"
    )
    if stale:
        print("  the ledger claims debt the import no longer owes:")
        for entry in stale:
            print(f"    {entry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
