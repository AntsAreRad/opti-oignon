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
exit status -- a clean end, an integer, a message -- is this process's. A
child process the guard starts is not covered, on purpose: the import
footprint guard measures in a fresh interpreter, and a firewall preloaded
there would change what it measures.

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
