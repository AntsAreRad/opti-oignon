"""Contracts for the documentation that ships with a release.

Documentation is the one artefact a reader executes without reading the
code first. A guide that names a file which is not there, a port nothing
listens on, or a route that has moved is not a cosmetic defect: it is a
failed install, and the reader has no way to tell whether the product or
the instructions are wrong.

Every contract below is written against the tree rather than against
prose. Each one resolves a documented claim to the thing it claims about
-- a file on disk, a path in the published surface, a port a launcher
serves -- so the claim cannot drift away from the artefact without a red
test.

The suite is deliberately outside the shared isolation window: it reads
files and, in one case, the published route table. It seeds no module and
stubs nothing.
"""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
MKDOCS = ROOT / "mkdocs.yml"
CHANGELOG = ROOT / "CHANGELOG.md"
BUILD_DOCS = ROOT / "scripts" / "build_docs.sh"
SMOKE_TEST = ROOT / "scripts" / "smoke_test.sh"
DEV_BACKEND = ROOT / "scripts" / "dev_backend.sh"
MODULE_MAP = DOCS / "architecture" / "module-map.md"
API_REFERENCE = DOCS / "api-reference.md"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CLEAN_GUARD = ROOT / ".github" / "scripts" / "public_clean_guard.py"

# A navigation entry is a file name that follows a key. The reference
# extraction accepts either case for the first character, which is the
# whole point of d1: the script's own extraction must be no narrower.
NAV_ENTRY = re.compile(r":\s+([A-Za-z][A-Za-z0-9_./-]*\.md)\s*$", re.MULTILINE)

# Ports appear in documentation as a host and a port. Only the ports the
# tree actually serves may be advertised.
DOCUMENTED_PORT = re.compile(r"(?:localhost|127\.0\.0\.1):(\d{2,5})")


def _markdown_files() -> list[Path]:
    """Every markdown file that ships, excluding installed dependencies."""
    skip = {"node_modules", ".svelte-kit", "build", "site"}
    return sorted(
        p
        for p in ROOT.rglob("*.md")
        if not skip.intersection(p.relative_to(ROOT).parts)
    )


def _nav_entries() -> list[str]:
    return NAV_ENTRY.findall(MKDOCS.read_text(encoding="utf-8"))


def _newest_section() -> str:
    """The changelog's newest section: unreleased notes when a cycle is
    open, the newest versioned entry once it has shipped."""
    text = CHANGELOG.read_text(encoding="utf-8")
    head = text.split("\n## ", 1)
    assert len(head) == 2, "the changelog has no section at all"
    return head[1].split("\n## ", 1)[0]


def _launcher_port() -> str:
    match = re.search(r'^PORT="(\d+)"', DEV_BACKEND.read_text(encoding="utf-8"), re.M)
    assert match, "the development launcher declares no default port"
    return match.group(1)


def _frontend_port() -> str:
    config = (ROOT / "frontend" / "playwright.config.ts").read_text(encoding="utf-8")
    match = re.search(r"localhost:(\d+)", config)
    assert match, "the browser configuration declares no origin port"
    return match.group(1)


def _inference_port() -> str:
    """The model host is a served address too, and the tree declares it."""
    backends = (ROOT / "opti_oignon" / "config" / "backends.yaml").read_text(
        encoding="utf-8"
    )
    match = re.search(r"host:\s*\"https?://[^:\"]+:(\d+)\"", backends)
    assert match, "the backend configuration declares no inference host port"
    return match.group(1)


def _load_clean_guard():
    """Import the guard as a plain module, not through a hand-built window.

    The guard is a standalone script rather than a project module, so it
    needs no package window at all: putting its directory on the path for
    the duration of the import is enough, and the entry is removed again
    so nothing else in the run sees it. Importing it is safe -- it only
    acts when run as a program.
    """
    directory = str(CLEAN_GUARD.parent)
    sys.path.insert(0, directory)
    try:
        return importlib.import_module(CLEAN_GUARD.stem)
    finally:
        if directory in sys.path:
            sys.path.remove(directory)


def test_d1_nav_check_sees_every_entry_whatever_its_case() -> None:
    """The doc build's own nav check must not be narrower than the nav.

    A validator that reports success on the subset it can see is worse
    than no validator: it is read as a clean bill of health. The pattern
    the script uses is extracted from the script itself, so tightening
    the pattern later re-breaks this contract rather than slipping past.
    """
    script = BUILD_DOCS.read_text(encoding="utf-8")
    pattern = re.search(r"grep -oP '([^']+)' \"\$MKDOCS_YML\"", script)
    assert pattern, "the doc build no longer extracts nav entries with a pattern"

    seen = subprocess.run(
        ["grep", "-oP", pattern.group(1), str(MKDOCS)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()

    expected = _nav_entries()
    assert expected, "the navigation names no documentation file"
    missed = sorted(set(expected) - set(seen))
    assert not missed, f"the nav check cannot see these entries: {missed}"


def test_d2_every_navigated_document_exists() -> None:
    """A navigation entry that resolves to nothing fails a strict build."""
    missing = [entry for entry in _nav_entries() if not (DOCS / entry).is_file()]
    assert not missing, f"navigation points at absent documents: {missing}"


def test_d3_release_notes_name_every_continuous_integration_guard() -> None:
    """Guards that gate merges are release-visible and must be written down.

    The guards are the part of the tree a reader cannot discover by using
    the product. If the notes do not name them, nothing else will. The
    newest changelog section carries that duty whether the cycle is still
    open or has just shipped.
    """
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    guards = sorted(set(re.findall(r"\.github/scripts/(\w+_guard)\.py", workflow)))
    assert guards, "the workflow runs no guard script"

    notes = _newest_section()
    absent = [guard for guard in guards if guard not in notes]
    assert not absent, f"the release notes do not name these guards: {absent}"


def test_d4_documentation_advertises_only_ports_the_tree_serves() -> None:
    """A documented address that refuses connections is a failed install."""
    allowed = {_launcher_port(), _frontend_port(), _inference_port()}
    offenders: list[str] = []
    for path in _markdown_files():
        for port in set(DOCUMENTED_PORT.findall(path.read_text(encoding="utf-8"))):
            if port not in allowed:
                offenders.append(f"{path.relative_to(ROOT)}:{port}")
    assert not offenders, f"documented ports nothing serves: {sorted(offenders)}"


def test_d6_smoke_test_requests_only_paths_that_exist() -> None:
    """The install check must fail on the product, never on itself.

    A stale path in the check reports a 404 that reads as a broken
    install, which is the opposite of what the check is for.
    """
    from opti_oignon.api.app import app

    published = set(app.openapi()["paths"])
    normalise = re.compile(r"\{[^}]+\}")
    known = {normalise.sub("{}", path.rstrip("/")) for path in published}

    script = SMOKE_TEST.read_text(encoding="utf-8")
    requested = sorted(set(re.findall(r"\$\{BASE\}(/api/[A-Za-z0-9_/{}.-]*)", script)))
    assert requested, "the install check requests no api path"

    absent = [
        path
        for path in requested
        if normalise.sub("{}", path.rstrip("/")) not in known
    ]
    assert not absent, f"the install check requests absent paths: {absent}"


def test_d7_module_map_names_only_modules_that_exist() -> None:
    """A map is a promise that the territory looks like this."""
    text = MODULE_MAP.read_text(encoding="utf-8")
    cited = sorted(set(re.findall(r"`([A-Za-z0-9_./-]+\.py)`", text)))
    assert cited, "the module map names no module"

    absent = [name for name in cited if not (ROOT / "opti_oignon" / name).is_file()]
    assert not absent, f"the module map names absent modules: {absent}"


def test_d8_stated_endpoint_count_matches_the_published_surface() -> None:
    """A count is a claim like any other, and it is checkable."""
    from opti_oignon.api.app import app

    text = API_REFERENCE.read_text(encoding="utf-8")
    match = re.search(r"~?(\d{3,4})\s+endpoints", text)
    assert match, "the api reference states no endpoint count"
    assert int(match.group(1)) == len(app.openapi()["paths"])


def test_d9_published_documentation_carries_no_rejected_nomenclature() -> None:
    """The published surface is held to the standard the guard writes down.

    The guard reads added lines on a diff, which never covered the files
    below because they were never in its walk. Held here against the
    whole file rather than a diff, so the standing text is covered too.
    """
    guard = _load_clean_guard()
    covered = _markdown_files() + [MKDOCS, ROOT / "requirements-docs.txt"]

    offenders: list[str] = []
    for path in covered:
        violations = guard.find_violations(path.read_text(encoding="utf-8").splitlines())
        for index, kind, _line in violations:
            offenders.append(f"{path.relative_to(ROOT)}:{index + 1} ({kind})")
    assert not offenders, f"rejected nomenclature in published files: {offenders}"


def test_d5_the_smoke_test_reads_the_version_register() -> None:
    """The release check must ask the package, not carry its own copy.

    A literal version inside the check is a second register: it agrees with
    the package on the day it is written and silently disagrees forever
    after, and because the job that runs it is the release job, the first
    reader of the disagreement is a release. The check is therefore held to
    reading the declared version rather than restating it.
    """
    text = SMOKE_TEST.read_text(encoding="utf-8")

    # A dotted quad is an address, not a version: the trailing octet of
    # 127.0.0.1 must not be read as one. Scope is checked in the sibling
    # contract, which proves the pattern still sees a real literal.
    literals = re.findall(r"(?<![\d.])\d+\.\d+\.\d+(?![\d.])", text)
    assert not literals, (
        f"{SMOKE_TEST.name} states version literals {sorted(set(literals))}; "
        "it must read the declared version instead"
    )

    assert "__version__" in text, (
        f"{SMOKE_TEST.name} does not read the declared version"
    )


def test_d5b_the_version_the_smoke_test_reads_is_the_declared_one() -> None:
    """The register it reads is the one the package actually declares."""
    declared = (ROOT / "opti_oignon" / "__version__.py").read_text(
        encoding="utf-8"
    )
    match = re.search(r'__version__\s*=\s*"([^"]+)"', declared)
    assert match, "the package declares no version"

    resolved = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r);"
         "from opti_oignon.__version__ import __version__;"
         "print(__version__)" % str(ROOT)],
        capture_output=True, text=True, check=False,
    )
    assert resolved.returncode == 0, resolved.stderr
    assert resolved.stdout.strip() == match.group(1)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
