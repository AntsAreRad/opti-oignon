#!/usr/bin/env python3
"""Contracts for the end-to-end test harness's own coherence.

The browser harness shipped complete -- a Playwright configuration, an
operator script, npm entry points -- and pointed at nothing. Its test
directory did not exist, one npm entry named a spec file nobody had
written, the directories a run drops were untracked by the ignore file,
and the operator script could not report a failure. Every one of those
reads as working from the outside, which is exactly what makes them
expensive: the harness looks wired, so no one checks. These clauses make
the wiring executable:

  * E1 -- the configured test directory exists. A harness aimed at an
    absent directory finds no tests and says so only when run.
  * E2 -- every npm entry point that names a spec file names one that
    exists. A dead path is a command that fails for a reason unrelated to
    the code under test.
  * E3 -- every spec in the tree lives under the configured directory. A
    spec outside it is never collected and never runs.
  * E4 -- the paths a run writes to are ignored by git. The report and
    artifact directories land inside the tree, so an unignored run turns
    the working copy dirty and can be committed by accident.
  * E5 -- the operator script can report a failure. Reading an exit status
    into a variable while errexit is armed is unreachable code: the shell
    has already left.
  * E6 -- the configured directory sits inside the package that owns the
    browser runner. A spec imports that runner, and the module lookup
    walks up from the spec's own directory, so a directory outside the
    package cannot resolve the import however correct the config reads.
    This one only shows itself the first time a spec is actually run,
    which is precisely when the directory is no longer empty.

Local-only. Runs under pytest or via the __main__ runner. The
configuration, the manifest and the operator script are parsed as text and
JSON; no application module and no node tooling is involved.
"""

import json
import re
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_CONFIG_PATH = REPO / "frontend" / "playwright.config.ts"
_MANIFEST_PATH = REPO / "frontend" / "package.json"
_RUNNER_PATH = REPO / "scripts" / "run_e2e.sh"
_IGNORE_PATH = REPO / ".gitignore"

# The package a spec imports its test functions from.
_RUNNER_PACKAGE = "@playwright/test"

# Directories never searched for specs: dependency and build output trees.
_SKIP_DIRS = {"node_modules", ".svelte-kit", "build", ".git", "__pycache__"}

# A quoted value assigned to a named key in the configuration.
_TEST_DIR = re.compile(r"testDir\s*:\s*['\"]([^'\"]+)['\"]")
_OUTPUT_FOLDER = re.compile(r"outputFolder\s*:\s*['\"]([^'\"]+)['\"]")
_OUTPUT_DIR = re.compile(r"outputDir\s*:\s*['\"]([^'\"]+)['\"]")

# Playwright's default directory for per-test artifacts, resolved against
# the directory holding the configuration file.
_DEFAULT_OUTPUT_DIR = "test-results"

# A command word that names a spec file rather than a flag or a verb.
_SPEC_ARGUMENT = re.compile(r"(?<![\w./-])[\w./-]+\.spec\.[jt]s(?![\w.])")

# Shell statements that arm or disarm errexit, and a capture of the last
# exit status into a variable.
_SET_STATEMENT = re.compile(r"^\s*set\s+([-+][a-zA-Z]+(?:\s+[-+a-zA-Z]+)*)")
_STATUS_CAPTURE = re.compile(r"^\s*(?:local\s+)?[A-Za-z_][A-Za-z0-9_]*=\$\?")


def _config_text():
    return _CONFIG_PATH.read_text(encoding="utf-8")


def _config_dir():
    return _CONFIG_PATH.parent


def _configured_test_dir():
    """Absolute path of the directory the configuration points at."""
    match = _TEST_DIR.search(_config_text())
    assert match, "the configuration must declare its test directory"
    return (_config_dir() / match.group(1)).resolve()


def _run_output_paths():
    """Absolute paths a run writes into, derived from the configuration.

    The HTML report folder is declared; the per-test artifact directory is
    Playwright's default unless the configuration overrides it.
    """
    text = _config_text()
    paths = []
    report = _OUTPUT_FOLDER.search(text)
    if report:
        paths.append((_config_dir() / report.group(1)).resolve())
    override = _OUTPUT_DIR.search(text)
    artifacts = override.group(1) if override else _DEFAULT_OUTPUT_DIR
    paths.append((_config_dir() / artifacts).resolve())
    return paths


def _ignore_entries():
    """Meaningful lines of the ignore file, normalised for comparison."""
    entries = set()
    for raw in _IGNORE_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        entries.add(line.strip("/"))
    return entries


def _is_ignored(path):
    """True when an ignore entry covers ``path`` or one of its parents."""
    entries = _ignore_entries()
    relative = path.relative_to(REPO)
    candidates = [relative] + list(relative.parents)
    for candidate in candidates:
        if candidate == Path("."):
            continue
        if candidate.as_posix() in entries:
            return True
        # A bare name with no separator matches at any depth.
        if candidate.name in entries and "/" not in candidate.name:
            return True
    return False


def _tree_specs():
    """Every spec file in the tree, dependency and build trees excluded."""
    found = []
    for path in REPO.rglob("*.spec.*"):
        if path.suffix not in {".ts", ".js"}:
            continue
        if _SKIP_DIRS & set(path.relative_to(REPO).parts):
            continue
        found.append(path)
    return found


# ---------------------------------------------------------------------------
# E1 -- the configured test directory exists
# ---------------------------------------------------------------------------
def test_e1_configured_test_directory_exists():
    target = _configured_test_dir()
    assert target.is_dir(), (
        f"the harness is aimed at {target.relative_to(REPO).as_posix()!r} "
        "and the tree does not carry it; the run reports no tests found, "
        "which reads the same as a suite that passed"
    )


# ---------------------------------------------------------------------------
# E2 -- every npm entry point naming a spec names a real one
# ---------------------------------------------------------------------------
def test_e2_npm_entry_points_name_real_specs():
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    scripts = manifest.get("scripts", {})
    e2e = {name: cmd for name, cmd in scripts.items() if "e2e" in name}
    assert e2e, "the manifest must expose at least one browser entry point"
    for name, command in sorted(e2e.items()):
        for argument in _SPEC_ARGUMENT.findall(command):
            target = (_config_dir() / argument).resolve()
            assert target.is_file(), (
                f"entry point {name!r} runs {argument!r} and the tree does "
                "not carry it; the command fails for a reason that has "
                "nothing to do with the code under test"
            )


# ---------------------------------------------------------------------------
# E3 -- every spec in the tree is collectable
# ---------------------------------------------------------------------------
def test_e3_every_spec_lives_under_the_configured_directory():
    target = _configured_test_dir()
    for spec in _tree_specs():
        assert target in spec.parents, (
            f"{spec.relative_to(REPO).as_posix()} sits outside the "
            "configured directory, so no run ever collects it"
        )


# ---------------------------------------------------------------------------
# E4 -- the directories a run writes are ignored
# ---------------------------------------------------------------------------
def test_e4_run_output_paths_are_ignored():
    for path in _run_output_paths():
        assert _is_ignored(path), (
            f"a run writes {path.relative_to(REPO).as_posix()!r} inside the "
            "tree and no ignore entry covers it; the first run dirties the "
            "working copy and the output can be committed by accident"
        )


# ---------------------------------------------------------------------------
# E5 -- the operator script can report a failure
# ---------------------------------------------------------------------------
def test_e5_runner_can_report_a_failure():
    errexit = False
    for number, line in enumerate(
        _RUNNER_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        statement = _SET_STATEMENT.match(line)
        if statement:
            flags = statement.group(1)
            for token in flags.split():
                if token.startswith("-") and "e" in token[1:]:
                    errexit = True
                elif token.startswith("+") and "e" in token[1:]:
                    errexit = False
            continue
        if _STATUS_CAPTURE.match(line) and errexit:
            raise AssertionError(
                f"{_RUNNER_PATH.name} line {number} reads an exit status "
                "while errexit is armed; the shell has already left on the "
                "failing command, so the branch reporting the failure is "
                "unreachable and the script cannot say a run went red"
            )


# ---------------------------------------------------------------------------
# E6 -- the specs can resolve the runner they import
# ---------------------------------------------------------------------------
def test_e6_test_directory_can_resolve_the_browser_runner():
    target = _configured_test_dir()

    owner = None
    for candidate in [target] + list(target.parents):
        manifest = candidate / "package.json"
        if not manifest.is_file():
            continue
        declared = json.loads(manifest.read_text(encoding="utf-8"))
        names = set(declared.get("dependencies", {}))
        names |= set(declared.get("devDependencies", {}))
        if _RUNNER_PACKAGE in names:
            owner = candidate
            break
        if REPO not in candidate.parents and candidate != REPO:
            break

    assert owner is not None, (
        f"no package declaring {_RUNNER_PACKAGE!r} sits at or above "
        f"{target.relative_to(REPO).as_posix()!r}; a spec there cannot "
        "resolve the runner it imports, and the failure only surfaces on "
        "the first run that finds a spec to collect"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("E1 configured test directory exists",
         test_e1_configured_test_directory_exists),
        ("E2 entry points name real specs",
         test_e2_npm_entry_points_name_real_specs),
        ("E3 every spec is collectable",
         test_e3_every_spec_lives_under_the_configured_directory),
        ("E4 run output paths are ignored",
         test_e4_run_output_paths_are_ignored),
        ("E5 runner can report a failure",
         test_e5_runner_can_report_a_failure),
        ("E6 specs can resolve the browser runner",
         test_e6_test_directory_can_resolve_the_browser_runner),
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
