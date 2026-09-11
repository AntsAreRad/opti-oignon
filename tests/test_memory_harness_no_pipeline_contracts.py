#!/usr/bin/env python3
"""Contract that the memory harness is measurement only: nothing on the chat
path imports it.

The memory block's first session builds the instrument and writes no
pipeline. That is a rule about the tree, and a rule about the tree is
checked on the tree: the three harness modules exist, and no other module in
the package imports any of them -- not the executor, not the agent, not the
routes, and not the memory package's own facade, which would pull them into
every import of the package.

  * ZP1 -- the three harness modules exist and are pure at module scope.
  * ZP2 -- no module outside the harness imports a harness module.

Local-only (the public distribution ships no tests). Reads the tree; loads
nothing.
"""

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO  # noqa: E402

_PACKAGE = REPO / "opti_oignon"
_HARNESS = ("probes", "drift", "baseline")
_HARNESS_PATHS = {_PACKAGE / "memory" / f"{m}.py" for m in _HARNESS}
_STDLIB_ONLY_AT_SCOPE = {"dataclasses", "re", "typing", "collections", "hashlib", "json", "math", "itertools", "logging"}


def _imports_of(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(("." * node.level) + (node.module or ""))
    return names


# ---------------------------------------------------------------------------
# ZP1 -- the harness exists and is pure at module scope
# ---------------------------------------------------------------------------
def test_zp1_the_harness_modules_exist_and_are_pure_at_module_scope():
    for path in sorted(_HARNESS_PATHS):
        assert path.is_file(), f"{path.name} is part of the harness"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        top = {
            n for n in _imports_of(ast.Module(body=[s for s in tree.body if isinstance(s, (ast.Import, ast.ImportFrom))], type_ignores=[]))
        }
        offenders = {n for n in top if n.split(".")[0] not in _STDLIB_ONLY_AT_SCOPE and not n.startswith("__future__")}
        assert offenders == set(), (
            f"{path.name} imports only the standard library at module scope; "
            f"anything heavier is imported inside the function that needs it: {offenders}"
        )


# ---------------------------------------------------------------------------
# ZP2 -- nothing outside the harness imports it
# ---------------------------------------------------------------------------
def test_zp2_nothing_on_the_chat_path_imports_the_harness():
    offenders = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        if path in _HARNESS_PATHS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in _imports_of(tree):
            bare = name.lstrip(".")
            last = bare.split(".")[-1] if bare else ""
            if (
                bare.startswith("opti_oignon.memory.") and last in _HARNESS
            ) or (name.startswith(".") and last in _HARNESS and "memory" in str(path)):
                offenders.append(f"{path.relative_to(REPO)} imports {name}")
    assert offenders == [], (
        "the harness is measurement, not pipeline: nothing imports it yet. "
        f"Found: {offenders}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
