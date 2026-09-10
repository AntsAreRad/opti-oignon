#!/usr/bin/env python3
"""Contracts for the script that drives the interlanguage round trip.

The bridge is two languages meeting at a name. The concordance checker proves
the two TEXTS agree; only a compiler and a running JVM prove the binding
holds. That proof needs a Kotlin compiler, a C++ compiler and a JVM, and it
therefore cannot run where these contracts run.

Which is the whole difficulty. A script that cannot run its measurement has
three honest options and one dishonest one. It may run and pass, run and
fail, or refuse to run and say what it is missing. What it may never do is
report success having measured nothing -- and that is the easy mistake,
because the absence of a failure looks like a pass from the outside.

So these contracts pin the SHAPE of the script rather than its result:

  * RT1 -- a missing tool is refused, by name, with an exit code of its own.
    Measured here for real: this machine has no Kotlin compiler.
  * RT2 -- the refusal is distinct from both success and failure. An unrun
    measurement is owed, not passed and not failed.
  * RT3 -- no path prints a success line without having run the round trip.
  * RT4 -- the sentinel asserted is not zero, because zero is success in both
    return conventions the bridge uses.
  * RT5 -- the pinned compiler version is named, and a different one is
    refused rather than silently accepted.
  * RT6 -- the script is import-safe, so these contracts can read it without
    running a compiler.
"""

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "native_bridge_roundtrip.py"

# Exit codes the script promises. OWED is the one that matters: it is neither
# a pass nor a failure, and nothing downstream may read it as either.
OK, FAILED, OWED = 0, 1, 2


_UNDER_CONTRACT = "_native_bridge_roundtrip_under_contract"


def _module():
    """Load the script through the shared window; returns (module, restore).

    The script is import-safe by contract -- RT6 -- but a suite that loads a
    module still goes through the shared window, and still closes it. A
    window left open outlives the contract that opened it and breaks whatever
    runs next.
    """
    loaded, restore = isolate(targets={_UNDER_CONTRACT: SCRIPT})
    return loaded[_UNDER_CONTRACT], restore


def test_rt1_a_missing_tool_is_refused_by_name():
    """Measured, not simulated: this machine has no Kotlin compiler."""
    module, restore = _module()
    try:
        missing = module.missing_tools()
        assert "kotlinc" in missing, (
            "the probe reports no missing tool on a machine that has none of "
            "them installed, so it has never been shown able to report one"
        )
        result = subprocess.run(
            [sys.executable, str(SCRIPT)], capture_output=True, text=True,
        )
        assert result.returncode == OWED, (
            f"expected the owed code {OWED}, got {result.returncode}"
        )
        assert "kotlinc" in result.stdout + result.stderr, (
            "the refusal must name what is missing, or the reader cannot act"
        )
    finally:
        restore()


def test_rt2_the_refusal_is_neither_a_pass_nor_a_failure():
    module, restore = _module()
    try:
        assert module.OWED not in (module.OK, module.FAILED), (
            "an unrun measurement must be distinguishable from both outcomes"
        )
        result = subprocess.run(
            [sys.executable, str(SCRIPT)], capture_output=True, text=True,
        )
        combined = (result.stdout + result.stderr).lower()
        assert "owed" in combined, (
            "the word matters: the runbook and the tracking both record this as "
            "owed, and the script is where that claim starts"
        )
    finally:
        restore()


def test_rt3_no_path_reports_success_without_running():
    text = SCRIPT.read_text(encoding="utf-8")
    # Scoped to main's body on purpose. Comparing against the first mention
    # of the round trip anywhere in the file compares against its DEFINITION,
    # which precedes everything and makes this contract vacuous. Its own
    # blade found that.
    start = text.index("def main(")
    body = text[start:text.index('if __name__ == "__main__"')]
    call = body.index("_run_round_trip(")
    for match in re.finditer(r"return OK", body):
        assert match.start() > call, (
            "a success is returned before the round trip has been run"
        )
    assert "return OK" in body, (
        "proven capable: there is a success to return at all"
    )


def test_rt4_the_sentinel_asserted_is_not_zero():
    module, restore = _module()
    try:
        assert module.SENTINEL != 0, (
            "zero is success in both return conventions the bridge uses; a round "
            "trip that asserted zero would pass on a stub that did nothing"
        )
        assert module.SENTINEL == -1, (
            "the bridge's not-implemented sentinel is -1; the harness must "
            "assert the same value the real stubs return"
        )
    finally:
        restore()


def test_rt5_the_pinned_version_is_named_and_enforced():
    module, restore = _module()
    try:
        assert module.KOTLIN_VERSION == "2.0.20", (
            "the pinned compiler version is part of the measurement; a runbook "
            "that does not say which compiler produced a number says nothing"
        )
        assert module.kotlin_version_ok(
            "kotlinc-jvm 2.0.20 (JRE 17.0.11+9)") is True
        assert module.kotlin_version_ok(
            "kotlinc-jvm 1.9.24 (JRE 17.0.11+9)") is False, (
            "a different compiler must be refused, not silently accepted"
        )
        assert module.kotlin_version_ok("") is False, (
            "and an unreadable version is not a match either"
        )
    finally:
        restore()


def test_rt6_the_script_is_import_safe():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "checkpoint_before_apply = True" in text
    guarded = text.index('if __name__ == "__main__"')
    for call in re.finditer(r"^main\(", text, re.M):
        assert call.start() > guarded, (
            "importing this script must not run a compiler"
        )


def test_rt7_the_runbook_names_the_gate_and_its_checks():
    """The gate is a claim about when evidence becomes possible.

    A gate named in a tracking document drifts from the tree that has to
    satisfy it. This one is named in the runbook, beside the commands that
    check it, and this contract keeps the two together.
    """
    runbook = (REPO / "android" / "BUILD_RUNBOOK.md").read_text(encoding="utf-8")
    assert "bound bridge" in runbook, (
        "the artefact that gates a two-device round must be named"
    )
    for check in ("native_bridge_concordance.py",
                  "native_bridge_roundtrip.py",
                  "gradlew :app:test"):
        assert check in runbook, f"the gate does not name {check}"
    # Scoped to the row, not the document. "sentinel" appears elsewhere in
    # this runbook, so searching the whole text would pass with the check
    # deleted -- which is exactly what its blade did before this line.
    rows = [one for one in runbook.splitlines()
            if one.startswith("| 3 |") and "sentinel" in one]
    assert rows, (
        "the check that separates a proven mechanism from a working bridge "
        "is that no entry point still returns the sentinel, and it must be "
        "a row of the gate rather than a remark somewhere above it"
    )


def test_rt8_the_workflow_names_the_skip():
    """CI checks the script's shape and does not run its measurement.

    Running it there would exit owed, which is not a failure and not a pass,
    and a job cannot express that. So CI does not run it -- and says so,
    because a measurement absent without a word reads as one that passed.
    """
    text = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    lines = [one for one in text.splitlines()
             if "native_bridge_roundtrip" in one and one.lstrip().startswith("#")]
    assert lines, (
        "the workflow must name the round trip it does not run, as a comment "
        "rather than a step: a step would fail the job on an owed code"
    )
