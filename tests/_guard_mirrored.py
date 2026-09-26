#!/usr/bin/env python3
"""Run one guard with the data places mirrored from HEAD, as CI sees them.

A guard that imports the application -- the published-prose guard builds the
OpenAPI schema from it -- runs the application's module-level singletons, and
those open the stores of the data places: the branch store through the
connection helper, so with the master key, the project vector store, the
presets. In CI the tree is a fresh checkout and nothing real is there. On the
maintainer's machine the ladder's guard tier did it on every run. Here the
guard runs in this process under the firewall the test session installs,
with a fresh mirror seeded from HEAD, and the real places are left as they
were.

The guard runs as ``python3 guard.py`` would run it: as ``__main__``, with its
own directory first on ``sys.path`` and its arguments on ``sys.argv``, and its
exit status -- a clean end, an integer, a message -- is this process's.
Run by the ladder, a child process the guard starts is not covered: this
file installs the firewall and carries it to no child. Inside a test
session this process is itself a covered child, and a guard's child that
inherits the environment is covered with it. The one guard that starts a
Python child, the import footprint guard, is not covered either way: it
replaces its child's ``PYTHONPATH`` with the tree's root, to measure a
fresh interpreter, and the child then never runs the session's hook. That
is a decision, not a necessity -- the hook loads nothing and would add one
module, itself, to the count the guard measures; covering it would take a
change to that guard, keeping the hook's directory on the path it replaces.
What it leaves open is exactly the regression that guard exists to catch:
a store opened at import would open the tree's real store once, in that
child, before the guard reported it. Measured under ``strace`` today, that
child reaches no data place, and a test session counts it among the
launches its firewall does not reach.

Usage: python3 tests/_guard_mirrored.py GUARD [ARGS...]
"""

import runpy
import sys
from pathlib import Path

import _data_firewall

ROOT = Path(__file__).resolve().parent.parent


def main(argv):
    if not argv:
        print("usage: _guard_mirrored.py GUARD [ARGS...]", file=sys.stderr)
        return 2
    guard = Path(argv[0]).resolve()
    firewall = _data_firewall.DataFirewall(ROOT)
    firewall.install()
    firewall.current = guard.name
    sys.argv = [str(guard), *argv[1:]]
    sys.path[0] = str(guard.parent)
    try:
        runpy.run_path(str(guard), run_name="__main__")
    finally:
        firewall.uninstall()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
