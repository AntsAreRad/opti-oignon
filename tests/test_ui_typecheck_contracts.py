#!/usr/bin/env python3
"""Contracts for the frontend type check: svelte-check errors, counted per file.

svelte-check runs through ``tests/_frontend.svelte_check_counts``: on a copy
of the frontend under ``$TMPDIR`` (the files the census lists, the
dependencies linked entry by entry), after ``svelte-kit sync``, with
``--tsconfig ./tsconfig.json --output machine``. Its errors are counted per
file and held by the ratchet engine (``check_ledger``) to a ledger born equal
to today's count, so a file never gains an error and a file that loses one
lowers the ledger. A single global total would let one file's growth hide
behind another file's repair; a per-file ledger does not.

  * UX14 -- svelte-check errors per file never exceed the ledger. Clauses:

      - held: a file never counts above its entry, and a file not in the
        ledger counts 0;
      - lowered: an entry above its file's count fails with "lower the
        ledger";
      - planted: an error planted in a component and one planted in a
        module, under ``src/lib`` and again under ``src/routes``, written
        into the copy and never into the tree, are each counted, so a check
        gone blind to either kind of file, or to either part of the source,
        reads red instead of a quiet zero; and svelte-check's own file total
        is at least the number of components and modules the census lists;
      - charged: every error svelte-check reports is charged to a file the
        census lists, so an error in a generated file, or in a kind of file
        the listing misses, is never dropped;
      - read: the helper asks for the machine output, whose summary has no
        singular form, and reads a single error as one;
      - refused: a run that prints no ``COMPLETED`` line (svelte-check died
        before its summary) is refused, never read as zero, and so is a run
        whose ``COMPLETED`` total differs from the ``ERROR`` lines read.

    The reading and refusal clauses run a stand-in svelte-check in a
    synthetic repository under ``$TMPDIR``; the others run the real one over
    the real tree.

Local-only (the public distribution ships no tests). Runs under pytest or
the __main__ runner. Needs Node and ``frontend/node_modules``; without them
the helper raises, and so does the contract.
"""

import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _frontend import check_ledger, files, svelte_check_counts  # noqa: E402

# Seconds each contract may take on this machine, read back by the ladder from the junit file.
BUDGET_S = {
    "test_ux14_svelte_check_errors_per_file_never_exceed_the_ledger": 15.0,
}

# The kinds of file svelte-check reports on: components, modules, scripts.
_CHECKED = (".svelte", ".ts", ".js", ".cjs", ".mjs")
_WITHIN = "frontend"

# Errors planted in the copy only, one in a component and one in a module.
_COMPONENT = "frontend/src/lib/typecheck_fixture.svelte"
_COMPONENT_TEXT = (
    '<script lang="ts">\n'
    "\tconst planted: number = 'a';\n"
    "</script>\n"
    "\n"
    "<p>{planted}</p>\n"
)
_MODULE = "frontend/src/lib/typecheck_fixture.ts"
_MODULE_TEXT = "export const planted: number = 'a';\n"
# The same two, planted under the routes.
_ROUTE_COMPONENT = "frontend/src/routes/typecheck_fixture.svelte"
_ROUTE_MODULE = "frontend/src/routes/typecheck_fixture.ts"
_PLANTED = {
    _COMPONENT: _COMPONENT_TEXT, _MODULE: _MODULE_TEXT,
    _ROUTE_COMPONENT: _COMPONENT_TEXT, _ROUTE_MODULE: _MODULE_TEXT,
}

# Variables that point git at another repository or index; a synthetic
# repository must answer for itself.
_GIT_LOCATORS = (
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_NAMESPACE",
)

_ERROR_LINE = (
    '1790000000001 ERROR "src/lib/a.svelte" 2:8 '
    '"Type \'string\' is not assignable to type \'number\'."\n'
)
# One error, as svelte-check prints it in each output.
_ONE_MACHINE = (
    '1790000000000 START "/stand-in/frontend"\n'
    + _ERROR_LINE
    + "1790000000002 COMPLETED 1 FILES 1 ERRORS 0 WARNINGS 1 FILES_WITH_PROBLEMS\n"
)
_ONE_HUMAN = (
    "Loading svelte-check in workspace: /stand-in/frontend\n"
    "svelte-check found 1 error and 0 warnings in 1 file\n"
)
# A run that died before its summary.
_DIED = '1790000000000 START "/stand-in/frontend"\n' + _ERROR_LINE
# A summary of two errors over one line the parser reads and one it cannot.
_SHORT = (
    '1790000000000 START "/stand-in/frontend"\n'
    + _ERROR_LINE
    + "1790000000001 ERROR src/lib/a.svelte 3:1 \"Cannot find name 'b'.\"\n"
    + "1790000000002 COMPLETED 1 FILES 2 ERRORS 0 WARNINGS 1 FILES_WITH_PROBLEMS\n"
)


def _tool(path, body):
    path.write_text("#!/bin/sh\n" + body, encoding="ascii")
    path.chmod(0o755)


def _stand_in(root, machine, human="", rc=1):
    """A synthetic repository whose svelte-check prints ``machine`` when asked
    for the machine output and ``human`` otherwise, then exits ``rc``."""
    component = root / "frontend" / "src" / "lib" / "a.svelte"
    component.parent.mkdir(parents=True)
    component.write_text("<p>a</p>\n", encoding="utf-8")
    (root / "machine.txt").write_text(machine, encoding="utf-8")
    (root / "human.txt").write_text(human, encoding="utf-8")
    tools = root / "frontend" / "node_modules" / ".bin"
    tools.mkdir(parents=True)
    _tool(tools / "svelte-kit", "exit 0\n")
    _tool(
        tools / "svelte-check",
        'case " $* " in\n'
        f'  *" --output machine "*) cat {shlex.quote(str(root / "machine.txt"))} ;;\n'
        f"  *) cat {shlex.quote(str(root / 'human.txt'))} ;;\n"
        "esac\n"
        f"exit {rc}\n",
    )
    env = {k: v for k, v in os.environ.items() if k not in _GIT_LOCATORS}
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    subprocess.run(
        ["git", "init", "-q", str(root)], check=True, capture_output=True, env=env,
    )
    return root


def _count(counts):
    def count(path, text):
        return counts.get(path, 0)

    return count


# ---------------------------------------------------------------------------
# UX14 -- svelte-check errors per file never exceed the ledger
# ---------------------------------------------------------------------------
def test_ux14_svelte_check_errors_per_file_never_exceed_the_ledger():
    with tempfile.TemporaryDirectory(prefix="oo_ux14_") as tmp:
        # read: the machine output is asked for, and one error reads as one.
        one = _stand_in(Path(tmp) / "one", _ONE_MACHINE, human=_ONE_HUMAN)
        assert svelte_check_counts(one) == {"frontend/src/lib/a.svelte": 1}

        # refused: a run with no COMPLETED line is never read as zero.
        died = _stand_in(Path(tmp) / "died", _DIED)
        with pytest.raises(AssertionError, match="printed no COMPLETED line"):
            svelte_check_counts(died)

        # refused: a total that differs from the ERROR lines read.
        short = _stand_in(Path(tmp) / "short", _SHORT)
        with pytest.raises(AssertionError, match="the parser missed some"):
            svelte_check_counts(short)

    counts = svelte_check_counts(extra=_PLANTED)

    # planted: each kind of file is still checked.
    assert counts.get(_COMPONENT, 0) >= 1, (
        f"the error planted in {_COMPONENT} was not counted: the check is "
        f"blind to components, and its zeros mean nothing"
    )
    assert counts.get(_MODULE, 0) >= 1, (
        f"the error planted in {_MODULE} was not counted: the check is blind "
        f"to modules (is the tsconfig passed?), and its zeros mean nothing"
    )
    missed = [path for path in (_ROUTE_COMPONENT, _ROUTE_MODULE) if counts.get(path, 0) < 1]
    assert not missed, (
        f"the errors planted under the routes were not counted: {missed}; "
        f"the check does not reach that part of the source"
    )
    sources = len(files((".svelte", ".ts")))
    assert counts.checked >= sources, (
        f"svelte-check says it checked {counts.checked} files, fewer than the "
        f"{sources} components and modules the census lists"
    )

    # charged: no error falls outside the census.
    listed = set(files(_CHECKED, within=_WITHIN))
    stray = sorted(set(counts) - listed - set(_PLANTED))
    assert not stray, (
        f"svelte-check charged errors to files the census does not list, so "
        f"the ledger cannot hold them: {stray}"
    )

    # held and lowered: the ratchet engine over the real tree.
    check_ledger(
        "UX14_LEDGER", UX14_LEDGER, _count(counts), (_COMPONENT, _COMPONENT_TEXT),
        test_file=__file__, suffixes=_CHECKED, within=_WITHIN,
    )


# ===========================================================================
# The ledger: svelte-check errors per file, born equal to today's count.
# ===========================================================================
UX14_LEDGER = {
    'frontend/src/lib/api/chat.ts': 2,
    'frontend/src/lib/components/chat/ChatMessage.svelte': 3,
    'frontend/src/lib/components/chat/ToolCallDisplay.svelte': 2,
    'frontend/src/lib/components/health/CacheManager.svelte': 1,
    'frontend/src/lib/components/panels/CompressionSettings.svelte': 1,
    'frontend/src/lib/components/panels/ProjectList.svelte': 1,
    'frontend/src/lib/components/panels/SandboxSettingsStrip.svelte': 5,
    'frontend/src/lib/components/panels/SyncPanel.svelte': 6,
    'frontend/src/lib/components/settings/FineTunePanel.svelte': 1,
    'frontend/src/lib/components/settings/PluginsPanel.svelte': 5,
    'frontend/src/lib/stores/chat.ts': 4,
}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
