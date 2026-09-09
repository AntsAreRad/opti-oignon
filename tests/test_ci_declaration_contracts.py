#!/usr/bin/env python3
"""Contracts for the CI workflow's self-declared header.

The workflow opens with an inventory of its jobs and states that the list
is a contract with the reader: a job added or removed below is a line
changed in the header in the same edit. Nothing enforced that, and the
header quietly fell behind the jobs it described. These clauses make the
stated contract executable:

  * C1 -- the header inventory names exactly the jobs the workflow
    declares: no job missing from the list, no listed job absent from the
    file.
  * C2 -- every guard script shipped under .github/scripts/ is wired into
    the workflow. A guard that exists but is not run protects nothing
    while looking like it does.
  * C3 -- every requirements file the workflow installs from exists in the
    tree. A dead reference swallowed by an error guard installs nothing
    and reads as a passing step.

Local-only. Runs under pytest or via the __main__ runner. The workflow is
parsed as text and YAML; no application module is imported.
"""

import re
import traceback
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
_CI_PATH = REPO / ".github" / "workflows" / "ci.yml"
_SCRIPTS_DIR = REPO / ".github" / "scripts"

# A header inventory line: comment marker, three spaces, a job id, then a
# dash-introduced description. Continuation lines indent deeper and do not
# match.
_HEADER_ENTRY = re.compile(r"^#   ([a-z][a-z0-9-]*)\s+- ")

# A requirements reference: pip's -r flag and the file it names. Anchored
# on the install verb so the recursive flag of other tools never matches.
_REQUIREMENTS_REF = re.compile(r"pip install\s+-r\s+([\w./-]+)")


def _ci_text():
    return _CI_PATH.read_text(encoding="utf-8")


def _header_inventory(text):
    """Job ids named by the leading comment block, in order."""
    names = []
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        match = _HEADER_ENTRY.match(line)
        if match:
            names.append(match.group(1))
    return names


def _declared_jobs(text):
    data = yaml.safe_load(text)
    return list(data["jobs"].keys())


# ---------------------------------------------------------------------------
# C1 -- the header inventory and the declared jobs are the same set
# ---------------------------------------------------------------------------
def test_c1_header_inventory_matches_declared_jobs():
    text = _ci_text()
    listed = _header_inventory(text)
    declared = _declared_jobs(text)
    assert listed, "the header inventory must exist and name the jobs"
    assert set(listed) == set(declared), (
        "the header calls itself a contract with the reader; a job the "
        f"list misses or invents breaks it (listed={sorted(listed)}, "
        f"declared={sorted(declared)})"
    )


# ---------------------------------------------------------------------------
# C2 -- every shipped guard script is wired into the workflow
# ---------------------------------------------------------------------------
def test_c2_every_shipped_guard_is_wired():
    text = _ci_text()
    scripts = sorted(path.name for path in _SCRIPTS_DIR.glob("*.py"))
    assert scripts, "the guard directory must not be empty"
    for name in scripts:
        assert name in text, (
            f"{name} ships with the workflow but no step runs it; a guard "
            "that exists unwired protects nothing while looking covered"
        )


# ---------------------------------------------------------------------------
# C3 -- every requirements file the workflow installs from exists
# ---------------------------------------------------------------------------
def test_c3_requirements_references_name_real_files():
    text = _ci_text()
    refs = _REQUIREMENTS_REF.findall(text)
    assert refs, "the workflow must install from at least one pinned file"
    for ref in refs:
        assert (REPO / ref).is_file(), (
            f"the workflow installs from {ref!r} but the tree does not "
            "carry it; the step's error guard swallows the failure and "
            "the job runs with nothing installed"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("C1 header inventory matches jobs",
         test_c1_header_inventory_matches_declared_jobs),
        ("C2 every shipped guard is wired",
         test_c2_every_shipped_guard_is_wired),
        ("C3 requirements references are real",
         test_c3_requirements_references_name_real_files),
    ]
    passed = 0
    for label, fn in tests:
        try:
            fn()
            print(f"PASS  {label}")
            passed += 1
        except Exception:  # noqa: BLE001 -- report and continue
            print(f"FAIL  {label}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
