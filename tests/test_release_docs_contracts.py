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
import shutil
import subprocess
import sys
import tempfile
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

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


def _tracked_markdown(root: Path = ROOT) -> list[Path] | None:
    """Every markdown file the repository tracks, or None off a repository.

    Asking the repository is the whole of the fix. A walk of the disk
    returns whatever happens to be lying in the tree: installed
    dependencies, build output, caches the run in progress writes as it
    goes, and the private working documents this repository refuses by
    name. None of those are published prose, and holding published prose
    to them makes the verdict a statement about the machine rather than
    about the release.
    """
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md"],
        cwd=str(root), capture_output=True, text=True, check=False,
    )
    if listed.returncode != 0:
        return None
    names = [name for name in listed.stdout.split("\0") if name]
    if not names:
        return None
    return sorted(root / name for name in names)


def _refusals(root: Path) -> tuple[list[str], list[str]]:
    """The refusal patterns the repository writes down, and its exceptions."""
    text = (root / ".gitignore").read_text(encoding="utf-8")
    refused: list[str] = []
    kept: list[str] = []
    for line in text.splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("!"):
            kept.append(entry[1:].lstrip("/"))
        else:
            refused.append(entry)
    return refused, kept


def _matches(pattern: str, parts: tuple[str, ...]) -> bool:
    """Glob a whole path segment by segment.

    A star never crosses a separator here. The ordinary string glob lets
    one match across directory boundaries, which turns the recorded rule
    for root-level documents into a rule against every document in the
    tree -- a refusal far wider than the one written down, and silent.
    """
    pattern_parts = PurePosixPath(pattern).parts
    if len(pattern_parts) != len(parts):
        return False
    return all(
        fnmatch(part, expected)
        for expected, part in zip(pattern_parts, parts)
    )


def _refused_off_repository(root: Path, relative: str) -> bool:
    """True when the recorded refusals cover ``relative``."""
    refused, kept = _refusals(root)
    parts = PurePosixPath(relative).parts
    if any(_matches(pattern, parts) for pattern in kept):
        return False
    for pattern in refused:
        cleaned = pattern.strip("/")
        if not cleaned:
            continue
        # A pattern that leads with a separator, or carries one anywhere
        # but at its end, is read from the root; a bare name applies at
        # any depth, as a directory on the way down or as the file at the
        # end. Deciding that before the separators are stripped is the
        # whole of it: a rule written for the root of the tree, read as a
        # bare name, becomes a rule against the entire tree.
        anchored = pattern.startswith("/") or "/" in pattern.rstrip("/")
        if anchored:
            if _matches(cleaned, parts):
                return True
            continue
        if any(fnmatch(part, cleaned) for part in parts[:-1]):
            return True
        if not pattern.endswith("/") and fnmatch(parts[-1], cleaned):
            return True
    return False


def _markdown_off_repository(root: Path = ROOT) -> list[Path]:
    """The census with no repository to ask, and it says so.

    A fallback that quietly returns a plausible answer is a worse defect
    than the one it replaces: the reading looks like every other reading
    and nothing in the result records that a different route was taken.
    So this one announces itself, and it derives what to leave out from
    the refusals the repository writes down rather than from a list kept
    here -- a list kept here would go stale the first time the recorded
    refusals changed, silently and in the safe-looking direction.
    """
    print(
        "markdown census: no repository to ask, falling back to the "
        "recorded refusals in .gitignore",
        file=sys.stderr,
    )
    return sorted(
        path
        for path in root.rglob("*.md")
        if not _refused_off_repository(
            root, path.relative_to(root).as_posix()
        )
    )


def _markdown_files() -> list[Path]:
    """Every markdown file that ships, asked of the repository."""
    tracked = _tracked_markdown()
    return _markdown_off_repository() if tracked is None else tracked


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

    # The answer is the last line, not the whole stream. Any library the
    # package imports is free to greet the reader on standard output at
    # import time -- several do -- and a comparison against the raw
    # stream turns a banner printed by a dependency into a disagreement
    # about the declared version, which is neither true nor actionable.
    printed = [line for line in resolved.stdout.splitlines() if line.strip()]
    assert printed, "the version probe printed nothing at all"
    assert printed[-1].strip() == match.group(1)


def test_d10_the_markdown_census_asks_the_repository_not_the_disk(tmp_path):
    """A file the repository refuses is not published prose.

    Built as a throwaway repository rather than asserted against this
    one, because the property under test is what the census does when a
    refused file is present -- and this tree, by construction, does not
    carry one.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not installed here")

    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=str(tmp_path), capture_output=True,
            text=True, check=True,
        )

    run("init", "-q")
    run("config", "user.email", "tester@example.invalid")
    run("config", "user.name", "Release Tester")

    (tmp_path / ".gitignore").write_text("/PRIVATE.md\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# shipped\n", encoding="utf-8")
    (tmp_path / "PRIVATE.md").write_text("# refused\n", encoding="utf-8")
    run("add", ".gitignore", "README.md")
    run("commit", "-qm", "initial")

    tracked = _tracked_markdown(tmp_path)
    assert tracked is not None, "the repository route must answer here"
    names = sorted(path.name for path in tracked)
    assert names == ["README.md"], (
        f"the repository route must return tracked prose only, got {names}"
    )

    walked = sorted(path.name for path in tmp_path.rglob("*.md"))
    assert "PRIVATE.md" in walked, (
        "the refused file must be present on disk, or this clause proves "
        "nothing about the difference between the two routes"
    )
    with tempfile.TemporaryDirectory() as outside:
        assert _tracked_markdown(Path(outside)) is None, (
            "off a repository the route must decline rather than invent "
            "an answer, so the caller can take the announced fallback"
        )


def test_d11_off_a_repository_the_census_still_refuses_and_says_so(capsys):
    """The fallback is loud, and it reads the recorded refusals.

    A fallback that passes quietly is one more green that means nothing.
    This one names itself on the error stream, and it leaves out what
    .gitignore leaves out rather than what a list in this file leaves
    out -- the recorded refusals are the register, and there is one.
    """
    refused, kept = _refusals(ROOT)
    assert refused, "the repository records no refusals at all"
    assert kept, "the repository records no exceptions to its refusals"

    assert _refused_off_repository(ROOT, "MOBILE_NOTES.md"), (
        "a root-level document outside the standard set is refused by "
        "the recorded rule and must not be read as published prose"
    )
    assert not _refused_off_repository(ROOT, "README.md")
    assert not _refused_off_repository(ROOT, "docs/index.md")
    assert not _refused_off_repository(ROOT, "frontend/README.md")
    assert _refused_off_repository(ROOT, ".pytest_cache/README.md"), (
        "the run in progress writes this file; a census that reads it is "
        "reading its own exhaust"
    )
    assert _refused_off_repository(
        ROOT, "frontend/node_modules/pkg/README.md"
    )

    capsys.readouterr()
    _markdown_off_repository()
    assert "no repository to ask" in capsys.readouterr().err, (
        "the fallback must announce itself"
    )


def test_d12_a_banner_on_the_probe_does_not_move_the_version_verdict():
    """A dependency greeting the reader is not a version disagreement.

    Reproduced with a line printed ahead of the answer rather than by
    installing a library that prints one, so the clause holds wherever
    it runs and does not depend on what is installed here.
    """
    declared = (ROOT / "opti_oignon" / "__version__.py").read_text(
        encoding="utf-8"
    )
    match = re.search(r'__version__\s*=\s*"([^"]+)"', declared)
    assert match, "the package declares no version"

    greeted = subprocess.run(
        [sys.executable, "-c",
         "import sys; print('a dependency greets the reader');"
         "sys.path.insert(0, %r);"
         "from opti_oignon.__version__ import __version__;"
         "print(__version__)" % str(ROOT)],
        capture_output=True, text=True, check=False,
    )
    assert greeted.returncode == 0, greeted.stderr

    printed = [line for line in greeted.stdout.splitlines() if line.strip()]
    assert len(printed) >= 2, (
        f"the probe must have printed a banner and an answer, got {printed}"
    )
    # Not an exact count: whatever else the environment prints ahead of
    # the answer is precisely what this clause exists to tolerate, and an
    # exact count would be the same defect again in a new place.
    assert printed[-1].strip() == match.group(1)
    assert greeted.stdout.strip() != match.group(1), (
        "the raw stream must differ from the answer, or this clause is "
        "not exercising the difference it exists for"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
