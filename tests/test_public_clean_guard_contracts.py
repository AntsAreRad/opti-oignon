#!/usr/bin/env python3
"""Contracts for the public-clean CI guard.

The guard scans the added lines of a diff over the published trees and fails
when a line introduces internal session nomenclature. These contracts pin
the pure detection helper (find_violations) so its behaviour is verified
independently of git:

  * G1 -- a session code (the letter S then two-to-four digits, standalone)
    on an added line is flagged.
  * G2 -- an internal document reference (a known prefix followed by a
    session code or the tracking marker) is flagged.
  * G3 -- legitimate lines are NOT flagged: an uppercase constant that
    merely starts with one of the prefixes, ordinary lowercase identifiers,
    and the exempt public product terms.
  * G4 -- an internal process word is flagged.
  * G5 -- the scan perimeter covers every published tree that ships, not
    the Python trees alone. The detector is a regex over added lines and is
    agnostic to language, so a tree of TypeScript, shell or Kotlin is
    guarded on exactly the same terms as a tree of Python.
  * G7 -- a lowercase session code is flagged: case is not a disguise.
  * G8 -- a session code continued by a capitalized word (the camel-case
    identifier shape) is flagged: a missing word boundary is not an exit.
  * G9 -- scope control: a hardware architecture triple, whose digits run
    straight into a lowercase letter, is NOT flagged. The widened pattern
    must catch disguises without charging ordinary platform names.
  * G10 -- a tracked file whose name carries a session code is flagged, and
    an ordinary name is not: a name ships as surely as a line does.
  * G11 -- a tracked path naming the tool used to write the tree is flagged,
    in any case and in any component.
  * G12 -- a content line naming that tool is flagged, in any case, while a
    line naming a model vendor's call format is not.
  * G13 -- the tree-wide pass charges a tracked file at the root, outside
    the diff's scan trees, for its name or for its content; a clean tree
    passes.
  * G14 -- outside the scan trees every tracked line is read, not the added
    ones alone: a code on a standing line of a file at the root, under the
    documentation, under the CI tree or under any other directory is
    charged.
  * G15 -- there the whole rule applies: a document reference and a process
    word are charged as well as a code.
  * G16 -- scope control: a standing line under a scan tree is NOT charged
    by the tree-wide pass; the standing debt there is left to the diff pass
    until it is paid.
  * G17 -- a tree git cannot read fails the guard: nothing read is never a
    pass.
  * G18 -- a base git cannot resolve fails the guard: a diff git could not
    produce is not an empty diff.
  * G19 -- a read of the tracked tree that git fails, the names or the tool
    mentions, fails the guard: half a read is not a pass.
  * G20 -- the green line says how many added lines the diff pass read.
  * G21 -- a markdown name in capitals that is not on the guard's closed
    list of public document names is flagged on a line; a public one, and a
    name not in capitals, is not.
  * G22 -- every name on that list is one the tree writes: a document it
    carries, or a name written elsewhere in it. The list cannot become a
    place to name private documents.
  * G23 -- a tracked file whose name is a document name in capitals off the
    list is flagged by name.
  * G24 -- the French word the internal numbering used for a block of work,
    capitalised as that numbering spelled it, is flagged; the same word in
    lower case, ordinary French for a block of code, and the English word
    are not.
  * G25 -- the tree-wide pass reads the scan trees as well: their standing
    debt is paid, and a standing line there is charged. It supersedes G16,
    which pinned the boundary while that debt was owed.

Every input that must be flagged is assembled from fragments at runtime, so
the literal nomenclature never appears in this file's source and the guard
does not trip on a scan of the test itself.

Local-only. Runs under pytest or via the __main__ runner. The guard script
lives under .github/, outside the importable package, and is loaded through
the shared isolation window.
"""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate  # noqa: E402

_GUARD = "_public_clean_guard_under_contract"
_GUARD_PATH = REPO / ".github" / "scripts" / "public_clean_guard.py"

# Fragments assembled at runtime; no clear nomenclature in the source text.
_S = "S"
_DIGITS = "312"
_UNDERSCORE = "_"
_TOOL = "Cla" + "ude"
_MD = "." + "md"


def _load():
    """Load the guard through the shared window; returns (module, restore)."""
    loaded, restore = isolate(targets={_GUARD: _GUARD_PATH})
    return loaded[_GUARD], restore


def _kinds(violations):
    return {kind for _idx, kind, _snippet in violations}


# ---------------------------------------------------------------------------
# G1 -- a standalone session code is flagged
# ---------------------------------------------------------------------------
def test_g1_session_code_is_flagged():
    guard, restore = _load()
    try:
        code = _S + _DIGITS  # assembled session code
        line = "    # delivered in " + code + " with contracts"
        violations = guard.find_violations([line])
        assert violations, "a standalone session code on an added line must flag"
        assert "session_code" in _kinds(violations)
    finally:
        restore()


# ---------------------------------------------------------------------------
# G2 -- an internal document reference is flagged
# ---------------------------------------------------------------------------
def test_g2_document_reference_is_flagged():
    guard, restore = _load()
    try:
        prefix = "PROMP" + "T"  # assembled prefix
        ref = prefix + _UNDERSCORE + _S + "313"  # doc reference form
        line = 'DOC = "' + ref + '.md"'
        violations = guard.find_violations([line])
        assert violations, "an internal document reference must flag"
        assert "doc_reference" in _kinds(violations)
    finally:
        restore()


# ---------------------------------------------------------------------------
# G3 -- legitimate lines are not flagged (incl. exempt product terms)
# ---------------------------------------------------------------------------
def test_g3_legitimate_lines_are_not_flagged():
    guard, restore = _load()
    try:
        # An uppercase constant that starts with a prefix but is NOT a doc ref.
        const_line = "SESSION" + _UNDERSCORE + "STATE_TOOLS = frozenset()"
        lowercase_line = "prompt_block = build_block(tools)"
        # Exempt public product terms must never account for a violation.
        exempt_line = 'permissions = ["' + "network" + _UNDERSCORE + 'outbound"]'
        injection_line = 'label = "' + "prompt" + _UNDERSCORE + 'injection"'
        for line in (const_line, lowercase_line, exempt_line, injection_line):
            assert guard.find_violations([line]) == [], (
                f"legitimate line must not be flagged: {line!r}"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# G4 -- an internal process word is flagged
# ---------------------------------------------------------------------------
def test_g4_process_word_is_flagged():
    guard, restore = _load()
    try:
        word = "back" + "fill"  # assembled process word
        line = "    # " + word + " the legacy rows"
        violations = guard.find_violations([line])
        assert violations, "an internal process word must flag"
        assert "process_word" in _kinds(violations)
    finally:
        restore()


# ---------------------------------------------------------------------------
# G5 -- the perimeter covers every published tree, not the Python ones only
# ---------------------------------------------------------------------------
def test_g5_perimeter_covers_every_published_tree():
    guard, restore = _load()
    try:
        paths = set(guard._SCAN_PATHS)
        for tree in ("opti_oignon/", "tests/", "frontend/", "scripts/",
                     "android/"):
            assert tree in paths, (
                f"the published tree {tree!r} must be inside the perimeter; "
                "the detector is language-agnostic, so leaving a shipped tree "
                "out is a choice, never a limitation"
            )
        # Every entry must name a tree that exists: a perimeter that lists a
        # path the repository does not carry guards nothing and reads clean.
        for tree in guard._SCAN_PATHS:
            assert (REPO / tree.rstrip("/")).is_dir(), (
                f"perimeter entry {tree!r} names no directory in the tree"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# G6 -- a lint pragma's rule code is exempt; a bare code elsewhere is not
# ---------------------------------------------------------------------------
def test_g6_lint_pragma_rule_code_is_exempt():
    guard, restore = _load()
    try:
        code = _S + _DIGITS  # assembled rule-code shape
        pragma_line = (
            "response = urlopen(request)  # noqa: " + code
            + " (scheme enforced above)"
        )
        assert guard.find_violations([pragma_line]) == [], (
            "a lint pragma names a rule of the linter, not a fragment of "
            "internal history; charging it would force authors to strip "
            "real suppressions to read clean"
        )
        # The exemption is the pragma form, never the bare token.
        bare_line = "    # see " + code + " for details"
        violations = guard.find_violations([bare_line])
        assert violations, "a bare standalone code outside a pragma must flag"
        assert "session_code" in _kinds(violations)
    finally:
        restore()


# ---------------------------------------------------------------------------
# G7 -- a lowercase session code is flagged
# ---------------------------------------------------------------------------
def test_g7_lowercase_session_code_is_flagged():
    guard, restore = _load()
    try:
        code = _S.lower() + _DIGITS  # assembled lowercase form
        for line in (
            "    # migrated in " + code + " with the queue",
            'strategy = "' + code + _UNDERSCORE + 'optimizer"',
            "route = '/api/cache/" + code + "/status'",
        ):
            violations = guard.find_violations([line])
            assert violations, f"lowercase form must flag: {line!r}"
            assert "session_code" in _kinds(violations)
    finally:
        restore()


# ---------------------------------------------------------------------------
# G8 -- a session code continued by a capitalized word is flagged
# ---------------------------------------------------------------------------
def test_g8_session_code_in_camel_case_identifier_is_flagged():
    guard, restore = _load()
    try:
        ident = _S + _DIGITS + "CacheStatusResponse"  # assembled identifier
        line = "class " + ident + "(BaseModel):"
        violations = guard.find_violations([line])
        assert violations, "the camel-case identifier shape must flag"
        assert "session_code" in _kinds(violations)
    finally:
        restore()


# ---------------------------------------------------------------------------
# G9 -- scope control: an architecture triple is not flagged
# ---------------------------------------------------------------------------
def test_g9_architecture_triple_is_not_flagged():
    guard, restore = _load()
    try:
        for line in (
            '"node_modules/@esbuild/linux-s390x": {',
            'arch = "s390x"',
        ):
            assert guard.find_violations([line]) == [], (
                f"a platform name must never be charged: {line!r}"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# G10 -- a session code in a tracked file's name is flagged
# ---------------------------------------------------------------------------
def test_g10_session_code_in_a_file_name_is_flagged():
    guard, restore = _load()
    try:
        name = "manifest" + _UNDERSCORE + _S + _DIGITS + "_full.md5"
        violations = guard.find_name_violations([name, "notes/" + name])
        assert [path for path, _kind in violations] == [name, "notes/" + name]
        assert {kind for _path, kind in violations} == {"session_code_in_name"}
        clean = ["scripts/ladder.sh", "frontend/node_modules/linux-s390x/a.js"]
        assert guard.find_name_violations(clean) == [], "an ordinary name is clean"
    finally:
        restore()


# ---------------------------------------------------------------------------
# G11 -- the tool's name in a tracked path is flagged, in any case
# ---------------------------------------------------------------------------
def test_g11_tool_name_in_a_path_is_flagged():
    guard, restore = _load()
    try:
        paths = [
            _TOOL.upper() + ".md",
            "." + _TOOL.lower() + "/rules/python.md",
            "docs/notes_" + _TOOL + "_setup.txt",
        ]
        violations = guard.find_name_violations(paths)
        assert [path for path, _kind in violations] == paths
        assert {kind for _path, kind in violations} == {"tool_in_name"}
    finally:
        restore()


# ---------------------------------------------------------------------------
# G12 -- the tool's name on a content line is flagged, in any case
# ---------------------------------------------------------------------------
def test_g12_tool_name_on_a_content_line_is_flagged():
    guard, restore = _load()
    try:
        lines = [
            "The local " + _TOOL + " milestone.",
            'cd "${' + _TOOL.upper() + '_PROJECT_DIR:-.}"',
            "." + _TOOL.lower() + "/",
            "- XML-style blocks -- a vendor-style call with parameters",
        ]
        assert [index for index, _snippet in guard.find_tool_mentions(lines)] == [0, 1, 2]
    finally:
        restore()


# ---------------------------------------------------------------------------
# G13 -- the tree-wide pass charges a root-level file, by name or content
# ---------------------------------------------------------------------------
def _place(root, files):
    for rel, text in files.items():
        target = Path(root) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)


def _tree_rc(guard, files, base="HEAD", added=None):
    """Run the guard's main in a fresh git tree holding ``files``.

    ``files`` are committed, so that ``HEAD`` names a base the diff pass can
    read; ``added`` are staged on top of that commit, the lines a change
    adds. A tree with no commit has no base at all, and a diff against it
    fails rather than coming back empty.
    """
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "init", "-q", tmp], check=True)
        _place(tmp, files)
        subprocess.run(["git", "-C", tmp, "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", tmp, "-c", "user.name=contract",
             "-c", "user.email=contract@example.invalid",
             "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
             "commit", "-q", "--no-verify", "-m", "base"],
            check=True,
        )
        if added:
            _place(tmp, added)
            subprocess.run(["git", "-C", tmp, "add", "-A"], check=True)
        os.chdir(tmp)
        try:
            return guard.main([base])
        finally:
            os.chdir(here)


def _tree_run(guard, files, base="HEAD", added=None):
    """``_tree_rc`` with what the guard printed: ``(rc, output)``."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = _tree_rc(guard, files, base=base, added=added)
    return rc, buffer.getvalue()


class _FailingGit:
    """Stands in for ``subprocess`` inside the guard: one git read fails."""

    def __init__(self, fails):
        self._fails = fails

    def run(self, cmd, **kwargs):
        if self._fails(cmd):
            empty = "" if kwargs.get("text") else b""
            return subprocess.CompletedProcess(cmd, 128, stdout=empty, stderr=empty)
        return subprocess.run(cmd, **kwargs)


def test_g13_tree_wide_pass_charges_a_root_level_file():
    guard, restore = _load()
    try:
        clean = {"README.md": "a clean tree\n"}
        assert _tree_rc(guard, clean) == 0, "a clean tree passes"
        named = dict(clean, **{"manifest_" + _S + _DIGITS + ".md5": "x\n"})
        assert _tree_rc(guard, named) == 1, "a root-level name is charged"
        mentioned = dict(clean, **{".gitignore": "." + _TOOL.lower() + "/\n"})
        assert _tree_rc(guard, mentioned) == 1, "a root-level mention is charged"
    finally:
        restore()


# ---------------------------------------------------------------------------
# G14 -- outside the scan trees, a code on a standing line is charged
# ---------------------------------------------------------------------------
def test_g14_tree_wide_pass_reads_every_line_outside_the_scan_trees():
    guard, restore = _load()
    try:
        clean = {"README.md": "a clean tree\n"}
        text = "a first line\nsee " + _S + _DIGITS + " for the history\n"
        for rel in ("CHANGELOG.md", "docs/notes.md", ".github/workflows/ci.yml",
                    "assets/credits.txt"):
            planted = dict(clean, **{rel: text})
            assert _tree_rc(guard, planted) == 1, f"a code in {rel} is charged"
    finally:
        restore()


# ---------------------------------------------------------------------------
# G15 -- outside the scan trees, the whole rule applies, not the code alone
# ---------------------------------------------------------------------------
def test_g15_tree_wide_pass_applies_the_whole_rule_outside_the_scan_trees():
    guard, restore = _load()
    try:
        clean = {"README.md": "a clean tree\n"}
        reference = "PROMP" + "T" + _UNDERSCORE + "TRACK" + "ING.md\n"
        word = "a " + "back" + "fill" + " of the rows\n"
        for text in (reference, word):
            planted = dict(clean, **{"CONTRIBUTING.md": text})
            assert _tree_rc(guard, planted) == 1, f"charged at the root: {text!r}"
    finally:
        restore()


# ---------------------------------------------------------------------------
# G16 -- the scan trees' standing lines are left to the diff pass
# ---------------------------------------------------------------------------
def test_g16_tree_wide_pass_leaves_the_scan_trees_to_the_diff_pass():
    guard, restore = _load()
    try:
        text = "# see " + _S + _DIGITS + "\n"
        for tree in guard._SCAN_PATHS:
            standing = {"README.md": "a clean tree\n", tree + "a.txt": text}
            assert _tree_rc(guard, standing) == 0, (
                f"a standing line under {tree} is the diff pass's business"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# G17 -- a tree git cannot read fails the guard instead of passing it
# ---------------------------------------------------------------------------
def test_g17_a_tree_git_cannot_read_fails_the_guard():
    guard, restore = _load()
    here, saved = os.getcwd(), os.environ.get("GIT_DIR")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["GIT_DIR"] = os.path.join(tmp, "absent")
            os.chdir(tmp)
            try:
                rc = guard.main(["HEAD"])
            finally:
                os.chdir(here)
        assert rc == 1, "nothing read must fail the guard, never pass it"
    finally:
        if saved is None:
            os.environ.pop("GIT_DIR", None)
        else:
            os.environ["GIT_DIR"] = saved
        restore()


# ---------------------------------------------------------------------------
# G18 -- a base git cannot resolve fails the guard
# ---------------------------------------------------------------------------
def test_g18_a_base_git_cannot_resolve_fails_the_guard():
    guard, restore = _load()
    try:
        clean = {"README.md": "a clean tree\n"}
        rc, out = _tree_run(guard, clean)
        assert rc == 0, f"the clean tree passes against a base git resolves: {out!r}"
        rc, out = _tree_run(guard, clean, base="refs/heads/no-such-base")
        assert rc == 1, "a diff git could not produce is not an empty diff"
        assert "no-such-base" in out, f"the failure names the base: {out!r}"
    finally:
        restore()


# ---------------------------------------------------------------------------
# G19 -- a tracked-tree read git fails fails the guard
# ---------------------------------------------------------------------------
def test_g19_a_tracked_tree_read_git_fails_fails_the_guard():
    guard, restore = _load()
    try:
        clean = {"README.md": "a clean tree\n"}
        reads = {
            "the tracked names": lambda cmd: cmd[1:2] == ["ls-files"],
            "the tool mentions": lambda cmd: cmd[1:2] == ["grep"] and "-i" in cmd,
        }
        for label, fails in reads.items():
            guard.subprocess = _FailingGit(fails)
            assert _tree_rc(guard, clean) == 1, f"a failed read of {label} fails the guard"
        guard.subprocess = subprocess
        assert _tree_rc(guard, clean) == 0, "the same tree, every read answered, passes"
    finally:
        restore()


# ---------------------------------------------------------------------------
# G20 -- the green says how many added lines the diff pass read
# ---------------------------------------------------------------------------
def test_g20_the_green_says_how_many_added_lines_it_read():
    guard, restore = _load()
    try:
        tree = guard._SCAN_PATHS[0]
        rc, out = _tree_run(guard, {"README.md": "a clean tree\n"},
                            added={tree + "notes.txt": "one\ntwo\nthree\n"})
        assert rc == 0, out
        assert "in 3 added line(s)" in out, f"the green carries its count: {out!r}"
    finally:
        restore()


# ---------------------------------------------------------------------------
# G21 -- a document name in capitals off the public list is flagged
# ---------------------------------------------------------------------------
def test_g21_a_capitalized_document_name_off_the_public_list_is_flagged():
    guard, restore = _load()
    try:
        charged = [
            "see " + "NOTES" + "_SYNC" + "_SPEC" + _MD + " for the record",
            "# taken from " + "PLAN" + _MD,
            "docs/" + "DESIGN" + "-" + "NOTES" + _MD,
        ]
        public = [
            "see README" + _MD + " and CHANGELOG" + _MD,
            "the skill lives in its SKILL" + _MD,
            "a lowercase notes" + _MD + " is not a name in capitals",
            # Three letters are an acronym's length, and the length of the
            # fragments these contracts build their references from.
            "an acronym such as FAQ" + _MD + " is not read as a document name",
        ]
        found = guard.find_violations(charged)
        assert [index for index, _kind, _snippet in found] == [0, 1, 2]
        assert _kinds(found) == {"doc_name"}
        assert guard.find_violations(public) == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# G22 -- every name on the public list is one the tree writes
# ---------------------------------------------------------------------------
def test_g22_every_public_document_name_is_one_the_tree_writes():
    guard, restore = _load()
    try:
        listed = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z"], capture_output=True, check=True,
        )
        carried = {Path(p).name for p in listed.stdout.decode("utf-8").split("\0") if p}
        assert "README" + _MD in carried, "control: the listing reads the tree"
        assert len(guard._PUBLIC_DOCS) >= 10
        for name in sorted(guard._PUBLIC_DOCS):
            if name in carried:
                continue
            found = subprocess.run(
                ["git", "-C", str(REPO), "grep", "-l", "-F", name, "--", ".",
                 ":(exclude).github/scripts/public_clean_guard.py",
                 ":(exclude)tests/test_public_clean_guard_contracts.py"],
                capture_output=True, text=True,
            )
            assert found.returncode == 0 and found.stdout.strip(), (
                f"{name} is on the public list and the tree never writes it"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# G23 -- a tracked file named in capitals off the public list is flagged
# ---------------------------------------------------------------------------
def test_g23_a_tracked_document_named_off_the_public_list_is_flagged():
    guard, restore = _load()
    try:
        internal = "docs/" + "ROAD" + "_NOTES" + _MD
        paths = [internal, "README" + _MD, "docs/guide" + _MD]
        assert guard.find_name_violations(paths) == [(internal, "doc_name_in_name")]
    finally:
        restore()


# ---------------------------------------------------------------------------
# G24 -- the capitalised French work-block word is flagged
# ---------------------------------------------------------------------------
def test_g24_the_capitalised_french_work_block_word_is_flagged():
    guard, restore = _load()
    try:
        word = "Bl" + "oc"
        charged = [
            "# " + word + " 3: the diff review",
            "SandboxPanel.svelte (Sandbox Workspace cycle, " + word + " 1)",
            '"""Resource Governor -- ' + word + "s 0-3: measurement.",
        ]
        free = [
            # Lower case is ordinary French for a block of code: the
            # language guard's business, not nomenclature.
            "# Extrait les " + word.lower() + "s de code d'un texte",
            "an English Block of text and a blocking call",
        ]
        found = guard.find_violations(charged)
        assert [index for index, _kind, _snippet in found] == [0, 1, 2]
        assert _kinds(found) == {"work_block"}
        assert guard.find_violations(free) == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# G25 -- the tree-wide pass reads the scan trees as well
# ---------------------------------------------------------------------------
def test_g25_tree_wide_pass_reads_the_scan_trees_too():
    guard, restore = _load()
    try:
        assert _tree_rc(guard, {"README.md": "a clean tree\n"}) == 0, "a clean tree passes"
        text = "# see " + _S + _DIGITS + "\n"
        for tree in guard._SCAN_PATHS:
            standing = {"README.md": "a clean tree\n", tree + "a.txt": text}
            assert _tree_rc(guard, standing) == 1, (
                f"a standing line under {tree} is charged: the debt there is paid"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("G1 session code flagged", test_g1_session_code_is_flagged),
        ("G2 document reference flagged", test_g2_document_reference_is_flagged),
        ("G3 legitimate lines pass", test_g3_legitimate_lines_are_not_flagged),
        ("G4 process word flagged", test_g4_process_word_is_flagged),
        ("G5 perimeter covers published trees",
         test_g5_perimeter_covers_every_published_tree),
        ("G6 lint pragma rule code exempt",
         test_g6_lint_pragma_rule_code_is_exempt),
        ("G7 lowercase session code flagged",
         test_g7_lowercase_session_code_is_flagged),
        ("G8 camel-case identifier shape flagged",
         test_g8_session_code_in_camel_case_identifier_is_flagged),
        ("G9 architecture triple not flagged",
         test_g9_architecture_triple_is_not_flagged),
        ("G10 session code in a file name flagged",
         test_g10_session_code_in_a_file_name_is_flagged),
        ("G11 tool name in a path flagged", test_g11_tool_name_in_a_path_is_flagged),
        ("G12 tool name on a content line flagged",
         test_g12_tool_name_on_a_content_line_is_flagged),
        ("G13 tree-wide pass charges a root-level file",
         test_g13_tree_wide_pass_charges_a_root_level_file),
        ("G14 every line read outside the scan trees",
         test_g14_tree_wide_pass_reads_every_line_outside_the_scan_trees),
        ("G15 whole rule applies outside the scan trees",
         test_g15_tree_wide_pass_applies_the_whole_rule_outside_the_scan_trees),
        ("G16 scan trees left to the diff pass",
         test_g16_tree_wide_pass_leaves_the_scan_trees_to_the_diff_pass),
        ("G17 unreadable tree fails the guard",
         test_g17_a_tree_git_cannot_read_fails_the_guard),
        ("G18 unresolvable base fails the guard",
         test_g18_a_base_git_cannot_resolve_fails_the_guard),
        ("G19 failed tracked-tree read fails the guard",
         test_g19_a_tracked_tree_read_git_fails_fails_the_guard),
        ("G20 green carries the added-line count",
         test_g20_the_green_says_how_many_added_lines_it_read),
        ("G21 unlisted document name flagged",
         test_g21_a_capitalized_document_name_off_the_public_list_is_flagged),
        ("G22 public names are written by the tree",
         test_g22_every_public_document_name_is_one_the_tree_writes),
        ("G23 unlisted document file name flagged",
         test_g23_a_tracked_document_named_off_the_public_list_is_flagged),
        ("G24 capitalised work-block word flagged",
         test_g24_the_capitalised_french_work_block_word_is_flagged),
        ("G25 tree-wide pass reads the scan trees",
         test_g25_tree_wide_pass_reads_the_scan_trees_too),
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
