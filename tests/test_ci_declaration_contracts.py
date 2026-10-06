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
  * C4 -- no step runs a contract the selection rule deselects. A step
    that overrides the rule (pytest -o addopts=...) drops every
    deselection for the files it names, and a contract superseded by name
    runs there as if it still held.

Local-only. Runs under pytest or via the __main__ runner. The workflow is
parsed as text and YAML, pyproject.toml as TOML; no application module is
imported.
"""

import re
import shlex
import traceback
from pathlib import Path

import tomllib
import yaml

REPO = Path(__file__).resolve().parents[1]
_CI_PATH = REPO / ".github" / "workflows" / "ci.yml"
_SCRIPTS_DIR = REPO / ".github" / "scripts"
_PYPROJECT = REPO / "pyproject.toml"

# A header inventory line: comment marker, three spaces, a job id, then a
# dash-introduced description. Continuation lines indent deeper and do not
# match.
_HEADER_ENTRY = re.compile(r"^#   ([a-z][a-z0-9-]*)\s+- ")

# A requirements reference: pip's -r flag and the file it names. Anchored
# on the install verb so the recursive flag of other tools never matches.
_REQUIREMENTS_REF = re.compile(r"pip install\s+-r\s+([\w./-]+)")

# A pytest option that replaces the selection rule: -o or --override-ini
# setting addopts. The rule's deselections stop applying to whatever that
# invocation collects.
_OVERRIDES_RULE = re.compile(r"(?:-o\s*|--override-ini[=\s]+)[\"']?addopts=")


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


def _deselected():
    """Node ids the selection rule deselects by name, split as pytest does."""
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    addopts = data["tool"]["pytest"]["ini_options"]["addopts"]
    args = shlex.split(addopts) if isinstance(addopts, str) else list(addopts)
    ids = []
    for index, arg in enumerate(args):
        if arg.startswith("--deselect="):
            ids.append(arg.split("=", 1)[1])
        elif arg == "--deselect" and index + 1 < len(args):
            ids.append(args[index + 1])
    return ids


def _pytest_runs(text):
    """(job, step name, command) for each step whose command runs pytest."""
    runs = []
    for job_id, job in yaml.safe_load(text)["jobs"].items():
        for step in job.get("steps", []):
            command = step.get("run") or ""
            if "pytest" in command:
                command = command.replace("\\\n", " ")
                runs.append((job_id, step.get("name", ""), command))
    return runs


def _collects(command, node_id):
    """Whether a pytest command collects node_id; no path named is the tree."""
    named = [t for t in shlex.split(command) if t.startswith("tests")]
    if not named:
        return True
    target = node_id.split("::")[0]
    for token in named:
        if "::" in token:
            if node_id == token or node_id.startswith(token + "["):
                return True
            continue
        path = token.rstrip("/")
        if target == path or target.startswith(path + "/"):
            return True
    return False


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
# C4 -- no step runs a contract the selection rule deselects
# ---------------------------------------------------------------------------
def test_c4_no_step_runs_a_contract_the_selection_rule_deselects():
    for override in ('pytest -o addopts="" tests',
                     "pytest --override-ini=addopts= tests"):
        assert _OVERRIDES_RULE.search(override), (
            f"control: the pattern must recognise {override!r}"
        )
    runs = _pytest_runs(_ci_text())
    deselected = _deselected()
    assert runs, "the workflow must run pytest in at least one step"
    assert deselected, (
        "the selection rule must read as deselecting contracts by name; an "
        "empty read would let every step pass unexamined"
    )
    for job_id, name, command in runs:
        if not _OVERRIDES_RULE.search(command):
            continue
        running = [node for node in deselected if _collects(command, node)]
        assert not running, (
            f"{job_id} / {name!r} overrides the selection rule and so runs "
            f"{len(running)} contract(s) it deselects by name, superseded "
            f"ones among them, as if they still held: {running[:3]}"
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
        ("C4 no step runs a deselected contract",
         test_c4_no_step_runs_a_contract_the_selection_rule_deselects),
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
