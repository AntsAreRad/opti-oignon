#!/usr/bin/env python3
"""What a fully-stubbed boot-check run reports, whatever ran before it.

Two of the boot-checklist contracts assert what a run reports when every
individual check has been replaced by a stub that passes. They are true
contracts, and they were carried by a harness that stubbed all but one
of the checks the run executes. The one left out reaches for a sibling
module at call time; when the sibling is unreachable it answers that it
could not tell and passes, and when the sibling is resident it answers
about the host and reports a warning. Whether it is resident depends on
which suites ran earlier in the same interpreter, so the verdict those
contracts read moved with sweep order and with which optional packages
the machine had installed -- never with the seam they were written to
pin. They passed alone and failed in a full run, and both readings were
about the wrong thing.

The two are superseded here rather than repaired in place: the suite
that holds them owes the isolation ledger, and its bytes may not move
without migrating it. They are deselected and re-asserted below against
a roster that is derived from the checklist itself rather than written
down beside it, so it cannot go stale when a ninth check is added, and
they are asserted twice over -- once with the sibling resident and once
with it absent -- so that agreement between the two readings is the
contract rather than an accident of ordering.

One thing the originals covered is not re-asserted and is not claimed:
they drove the real application lifespan, and these run inside the
shared isolation window. What a boot through the ASGI lifespan does is
out of reach here; what a fully-stubbed run reports, and what it leaves
in the cache the API endpoint reads, is not.

No server, no model, no network, and no module from outside the window
is ever consulted.
"""

import ast
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_CHECKLIST = "opti_oignon.startup_checks"
_CHECKLIST_FILE = source("startup_checks.py")
_RUNNER = "run_startup_checks"
_PREFIX = "_check_"


# ---------------------------------------------------------------------------
# Deriving the roster from the checklist, by two independent routes
# ---------------------------------------------------------------------------

def _executed_by_source():
    """Check names the runner calls, read by parsing the source."""
    tree = ast.parse(_CHECKLIST_FILE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == _RUNNER:
            return {
                call.func.id
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id.startswith(_PREFIX)
            }
    raise AssertionError(
        f"the checklist no longer defines {_RUNNER}; the roster this suite "
        "installs cannot be derived"
    )


def _defined_on_module(mod):
    """Check names the loaded module actually carries."""
    return {
        name for name in dir(mod)
        if name.startswith(_PREFIX) and callable(getattr(mod, name))
    }


def _sibling_modules():
    """Stand-ins for what a real check reaches for at call time.

    They answer the way an ordinary host answers -- an active backend the
    provenance gate does not cover -- which is the shape that makes an
    unstubbed check report a warning rather than stay silent.
    """
    provenance = types.ModuleType("opti_oignon.model_provenance")
    provenance.backend_enforces_provenance = lambda backend: False
    provenance.PROVENANCE_GATED_BACKENDS = frozenset({"llama_cpp"})

    mode = types.ModuleType("opti_oignon.security_mode")
    mode._default_backend = lambda: "ollama"
    mode.get_current_mode = lambda: "daily"
    return {
        "opti_oignon.model_provenance": provenance,
        "opti_oignon.security_mode": mode,
    }


class _Run:
    """A fully-stubbed checklist run, held open for inspection."""

    def __init__(self, *, siblings_resident):
        self._loaded, self._restore = isolate(
            targets={_CHECKLIST: _CHECKLIST_FILE},
            seeded=_sibling_modules() if siblings_resident else {},
        )
        self.mod = self._loaded[_CHECKLIST]
        self.roster = _executed_by_source()
        self._saved = {}
        for name in self.roster:
            self._saved[name] = getattr(self.mod, name)
            setattr(self.mod, name, lambda _n=name: self.mod.CheckItem(
                name=_n, passed=True, severity="info", detail="stub pass",
            ))
        self.mod.clear_cache()

    def unstubbed(self):
        """Check attributes the roster did not replace."""
        return {
            name for name in _defined_on_module(self.mod)
            if name not in self._saved
        }

    def close(self):
        for name, original in self._saved.items():
            setattr(self.mod, name, original)
        self.mod.clear_cache()
        self._restore()


def _both_residencies():
    for resident in (True, False):
        run = _Run(siblings_resident=resident)
        try:
            yield resident, run
        finally:
            run.close()


# ---------------------------------------------------------------------------
# b1 -- the derived roster is real: non-empty, and it matches the module
# ---------------------------------------------------------------------------

def test_b1_the_derived_roster_matches_what_the_module_carries():
    run = _Run(siblings_resident=False)
    try:
        assert run.roster, (
            "no check was derived from the runner: every assertion below "
            "would then be about a run that stubbed nothing"
        )
        assert run.roster == _defined_on_module(run.mod), (
            "the checks the runner calls and the checks the module defines "
            f"have diverged; called but undefined: "
            f"{sorted(run.roster - _defined_on_module(run.mod))}; defined "
            f"but never called: {sorted(_defined_on_module(run.mod) - run.roster)}"
        )
    finally:
        run.close()


# ---------------------------------------------------------------------------
# b2 -- no check escapes the roster the run installs
# ---------------------------------------------------------------------------

def test_b2_the_roster_leaves_no_check_running_for_real():
    for resident, run in _both_residencies():
        assert not run.unstubbed(), (
            f"with the sibling {'resident' if resident else 'absent'}, these "
            f"checks ran for real inside a fully-stubbed run: "
            f"{sorted(run.unstubbed())}"
        )


# ---------------------------------------------------------------------------
# b3 -- a fully-stubbed run passes, and reports the same either way
# ---------------------------------------------------------------------------

def test_b3_a_fully_stubbed_run_passes_in_both_residencies():
    verdicts = {}
    for resident, run in _both_residencies():
        result = run.mod.run_startup_checks()
        verdicts[resident] = result.all_passed
        assert result.all_passed, (
            f"a run whose every check was stubbed passing did not pass with "
            f"the sibling {'resident' if resident else 'absent'}: a check "
            "outside the roster ran for real, so the verdict follows sweep "
            "order rather than the stubs"
        )
    assert verdicts[True] == verdicts[False], (
        "the verdict of a fully-stubbed run depends on whether a sibling "
        "module happens to be loaded"
    )


# ---------------------------------------------------------------------------
# b4 -- and it never blocks
# ---------------------------------------------------------------------------

def test_b4_a_fully_stubbed_run_never_blocks():
    for resident, run in _both_residencies():
        result = run.mod.run_startup_checks()
        assert not result.blocked, (
            f"a fully-stubbed passing run blocked the boot with the sibling "
            f"{'resident' if resident else 'absent'}: {result.block_reason}"
        )


# ---------------------------------------------------------------------------
# b5 -- the cache the API endpoint reads carries that same verdict
# ---------------------------------------------------------------------------

def test_b5_the_cached_result_reports_the_run_that_produced_it():
    for resident, run in _both_residencies():
        result = run.mod.run_startup_checks()
        cached = run.mod.get_cached_result()
        assert cached is not None, (
            "a completed run left the cache empty; the API endpoint would "
            "re-run the checklist instead of serving the boot-time verdict"
        )
        assert cached.all_passed == result.all_passed, (
            "the cached verdict disagrees with the run that produced it"
        )
        assert cached.blocked == result.blocked, (
            "the cached blocked flag disagrees with the run that produced it"
        )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
