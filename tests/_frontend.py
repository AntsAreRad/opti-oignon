#!/usr/bin/env python3
"""Shared helpers for the frontend contracts: listing, reading, Node, svelte-check, ratchets.

A helper module, not a suite: it holds no contract of its own. The frontend
contracts are Python tests that read sources as text, run a dependency-free
TypeScript module under Node, or count what a tool reports. They prove a form
(declared, wired, counted, equal); they never prove behaviour in a browser.

What it provides:

  * ``files(suffixes)`` -- the listing rule every census uses,
    ``git ls-files --cached --others --exclude-standard frontend/src``, so a
    file being written and not yet staged is counted and an ignored one is
    not. An empty list raises: a census over nothing reads a false zero.
  * ``read(path)`` -- a repository file as UTF-8 text.
  * ``pattern_count(pattern)`` -- a count function over one regular
    expression, the shape ``check_ledger`` takes.
  * ``check_ledger(name, ledger, count_fn, fixture)`` -- the ratchet engine
    (rules below), proven by the RX contracts.
  * ``run_ts(modules, driver, clause)`` -- runs a dependency-free TypeScript
    module under Node's type stripping (node >= 22.6), with a driver that
    prints ``PASS <clause>``. Without Node it raises.
  * ``ssr()`` -- the session's server renderer: one vite server on a copy
    under ``$TMPDIR``, the ``$app`` modules stubbed by
    ``tests/_frontend_stubs/``; ``render(path, props)`` returns what a
    component emits when compiled for the server (see its docstring).
  * ``frontend_copy(dest)`` -- a working copy of ``frontend/`` (the listed
    files, the dependencies linked one package at a time), where a tool
    may write without touching the tree; the ladder builds and lints on it.
  * ``svelte_check_counts()`` -- svelte-check errors per file, on a copy of
    ``frontend/`` under ``$TMPDIR``, with files planted on request.
  * ``HELD`` -- the files held to the surface rules, and ``held_files()``,
    which expands its directory entries.

The ratchet rules. A ledger is a literal ``{path: count}`` dict assigned to a
module-level name in the test file that checks it.

  1. No file counts above its entry.
  2. A file not in the ledger counts 0.
  3. An entry above the current count is stale and fails with "lower the
     ledger"; so is an entry whose file is gone.
  4. No entry is higher than the same entry at ``HEAD``, read by
     ``git show HEAD:<test file>`` and ``ast``, never from the working tree.
     A path new since ``HEAD`` is admitted only as a rename that
     ``git diff -M HEAD`` shows, carrying its old entry and no more. The
     rename reader sees what the listing sees: the listing's untracked files
     are marked intent-to-add in a scratch copy of the index (never the
     index itself), so a move made without ``git mv`` is a rename when git's
     own similarity pairs it, and a new file is never one.
     A ledger that is not at ``HEAD`` under its name in its test file is
     looked for among the ledgers ``HEAD`` holds under the same directory:
     one that shares its files and is gone from the working tree (renamed,
     or moved to another test file) is its predecessor, and the ledger is
     compared with it; several such is refused. With no predecessor the
     ledger is being born and the census says so (``head == "born"``);
     rules 1 and 3 then hold it to today's exact count. What this cannot
     see: a predecessor kept in the file as an unchecked decoy while a new
     name carries the census.
  5. The file list is never empty.
  6. Every census carries a standing positive fixture, an assembled sample
     its count function must count at least once, so a probe that goes blind
     turns red instead of reading a false zero.

Rule 4 compares with ``HEAD``, so it holds between an edit and its commit;
on a clean checkout the ledger and ``HEAD`` are the same file.

Every finding of one census is reported together, in one failure.

It never builds an import window of its own. A contract that needs a guard
function loads it through ``tests/_isolation.isolate``.

Local-only (the public distribution ships no tests).
"""

from __future__ import annotations

import ast
import atexit
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parent.parent
SOURCE = "frontend/src"
NODE_MIN = (22, 6)

# Variables that point git at another repository or index. The helpers pass
# the repository explicitly and must not be redirected behind their back.
_GIT_LOCATORS = (
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_NAMESPACE",
)


HELD: list[str] = [
    "frontend/src/lib/ds/",
    "frontend/src/routes/dev/components/+page.svelte",
    "frontend/src/lib/nav/",
    "frontend/src/params/workshopSection.ts",
    "frontend/src/lib/components/layout/AppShell.svelte",
    "frontend/src/lib/components/layout/Sidebar.svelte",
    "frontend/src/lib/components/layout/StatusCard.svelte",
    "frontend/src/lib/components/layout/StopAllButton.svelte",
    "frontend/src/lib/components/layout/ApprovalsPill.svelte",
    "frontend/src/lib/components/layout/SpaceSwitch.svelte",
    "frontend/src/lib/components/layout/WorkshopBand.svelte",
    "frontend/src/lib/components/layout/PhoneHeader.svelte",
    "frontend/src/lib/components/settings/SettingsHub.svelte",
    "frontend/src/lib/stores/estop.ts",
    "frontend/src/lib/stores/backendStatus.ts",
    "frontend/src/lib/stores/exportDialog.ts",
    "frontend/src/lib/stores/approvals.ts",
    "frontend/src/lib/stores/lastRoutes.ts",
    "frontend/src/routes/+page.ts",
    "frontend/src/routes/(app)/+layout.svelte",
    "frontend/src/routes/(app)/+error.svelte",
    "frontend/src/routes/(app)/(use)/+layout.svelte",
    "frontend/src/routes/(app)/(use)/chat/+layout.svelte",
    "frontend/src/routes/(app)/(use)/preferences/+page.svelte",
    "frontend/src/routes/(app)/(workshop)/+layout.svelte",
    "frontend/src/routes/(app)/(workshop)/workshop/[section=workshopSection]/+page.svelte",
    "frontend/src/routes/benchmark/+page.ts",
    "frontend/src/routes/claims/+page.ts",
    "frontend/src/routes/health/+page.ts",
    "frontend/src/routes/settings/+page.ts",
    "frontend/src/routes/verify/+page.ts",
    "frontend/src/routes/verify-answer/+page.ts",
    "frontend/src/routes/verify-citations/+page.ts",
    "frontend/src/routes/+layout.svelte",
    "frontend/src/routes/(app)/(use)/chat/+page.svelte",
    "frontend/src/lib/chat/chatsIndex.ts",
    "frontend/src/lib/settings/search.ts",
    "frontend/src/lib/settings/catalog.ts",
    "frontend/src/lib/components/ui/ThemeSwitcher.svelte",
    "frontend/src/lib/components/ui/UserMenu.svelte",
    "frontend/src/lib/components/ui/NotificationCenter.svelte",
    "frontend/src/lib/components/settings/NetworkReachability.svelte",
    "frontend/src/lib/stores/estopState.ts",
    "frontend/src/routes/(app)/[...missing]/+page.ts",
    "frontend/src/lib/palette/",
    "frontend/src/lib/components/palette/",
    "frontend/src/lib/stores/palette.ts",
    "frontend/src/lib/components/ui/KeyboardShortcuts.svelte",
    "frontend/src/lib/stores/shortcutsHelp.ts",
    "frontend/src/lib/stores/shortcutKeys.ts",
    "frontend/src/lib/stores/notificationCenter.ts",
]
"""Files held to the surface rules: separation by tone and space rather than
by lines, matte surfaces, sentence-case labels, selection never by colour
alone. A file joins when it is rebuilt under those rules, and the list only
grows. An entry is a repository path, or a directory ending in ``/`` that
holds every listed file under it, the ones written later included
(``held_files`` expands it). The primitives come first: every surface is
drawn from them; their gallery beside them. Then the shell of both spaces:
the navigation table and its modules, the sidebar, the status card and the
stop, the phone header, the settings hub, the stores they read, the root
and route groups' layouts and the redirects of the old addresses; the chats
index and the module that builds its requests; the settings search and
the catalog it reads; what the old header held, moved into Preferences (the
palette switcher, the account menu, the notification history); the network
page's reachability; the stop's pure read rules; the catch-all that
answers an address no page serves; and the command palette: its modules
(its ranking, its registry of commands, its sources, its runner, what its
list says), its dialog and its store, with the shortcut handler that runs
the same commands, the store of its list of shortcuts, the store of the
keys each command runs by, and the store of the notification history's
panel, which one of those commands opens."""


def held_files(held=None, *, root=REPO):
    """The files the surface rules hold, sorted: every file ``held``
    (``HELD`` by default) names, a directory entry standing for every file
    the listing rule lists under it. Raises when the list is empty, or when
    an entry names nothing: a rule held over nothing reads a false zero."""
    held = HELD if held is None else held
    if not held:
        raise AssertionError(
            "HELD is empty: no file is held to the surface rules, and a rule "
            "held over nothing reads a false zero"
        )
    found = set()
    for entry in held:
        if entry.endswith("/"):
            found.update(files(None, within=entry.rstrip("/"), root=root))
        elif (Path(root) / entry).is_file():
            found.add(entry)
        else:
            raise AssertionError(f"HELD names {entry}, which is not a file of the tree")
    return sorted(found)


def _git(root, *args, index=None, stdin=None, ok=(0,)):
    """Run git in ``root``; any exit outside ``ok`` raises, with git's words.

    It never writes the repository's index: ``index`` names a scratch index
    file for the one call that marks files intent-to-add, and every other
    call is a read.
    """
    env = {k: v for k, v in os.environ.items() if k not in _GIT_LOCATORS}
    # A read must never refresh the index as a side effect.
    env["GIT_OPTIONAL_LOCKS"] = "0"
    if index is not None:
        env["GIT_INDEX_FILE"] = str(index)
    proc = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, env=env,
        input=stdin,
    )
    if proc.returncode not in ok:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {root} (rc={proc.returncode}): "
            f"{proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    return proc.stdout


def _split_z(raw):
    return [part for part in raw.decode("utf-8").split("\0") if part]


def _excluded(path, exclude):
    for entry in exclude:
        if entry.endswith("/"):
            if path.startswith(entry):
                return True
        elif path == entry:
            return True
    return False


def files(suffixes, *, within=SOURCE, exclude=(), root=REPO):
    """Repository-relative paths under ``within`` ending in one of ``suffixes``.

    The listing is ``git ls-files --cached --others --exclude-standard``: an
    untracked file is listed unless it is ignored. A path still in the index
    but gone from the disk is not. ``suffixes`` None lists every kind of
    file. ``exclude`` holds directory prefixes (ending in ``/``) and exact
    paths. Raises when the list is empty.
    """
    if isinstance(suffixes, str):
        suffixes = (suffixes,)
    suffixes = None if suffixes is None else tuple(suffixes)
    root = Path(root)
    raw = _git(
        root, "ls-files", "-z", "--cached", "--others", "--exclude-standard",
        "--", within,
    )
    listed = sorted(
        path for path in set(_split_z(raw))
        if (suffixes is None or path.endswith(suffixes))
        and not _excluded(path, exclude)
        and (root / path).is_file()
    )
    if not listed:
        kinds = "listed" if suffixes is None else ", ".join(suffixes)
        raise AssertionError(
            f"the file list is empty: no {kinds} file under "
            f"{within} in {root}; a census over nothing reads a false zero"
        )
    return listed


def read(path, *, root=REPO):
    """A repository file (relative path) as UTF-8 text."""
    return (Path(root) / path).read_text(encoding="utf-8")


def pattern_count(pattern, flags=0):
    """A count function: the matches of one regular expression in a file."""
    compiled = re.compile(pattern, flags)

    def count(path, text):
        return sum(1 for _ in compiled.finditer(text))

    count.pattern = compiled.pattern
    return count


@dataclass(frozen=True)
class Census:
    """What one ratchet read: the listing, the non-zero counts, and whether
    the ledger was compared with HEAD or is being born."""

    name: str
    files: tuple
    counts: dict
    head: str
    fixture: int
    compared_with: str = ""

    @property
    def total(self):
        return sum(self.counts.values())


def _module_values(source):
    """``{name: value node}`` for every module-level single-name assignment
    (the last one wins, as at import)."""
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                values[target.id] = node.value
    return values


def _ledger_at_head(root, test_file, name):
    """The ledger ``name`` as committed at HEAD, or None when HEAD has no
    ledger of that name in that test file."""
    rel = Path(test_file).resolve().relative_to(Path(root).resolve()).as_posix()
    _git(root, "rev-parse", "--verify", "--quiet", "HEAD^{commit}")
    if not _split_z(_git(root, "ls-tree", "-z", "--name-only", "HEAD", "--", rel)):
        return None
    source = _git(root, "show", f"HEAD:{rel}").decode("utf-8")
    found = _module_values(source).get(name)
    if found is None:
        return None
    try:
        value = ast.literal_eval(found)
    except ValueError as exc:
        raise AssertionError(f"{name} at HEAD in {rel} is not a literal dict") from exc
    if not isinstance(value, dict):
        raise AssertionError(f"{name} at HEAD in {rel} is not a dict")
    return value


def _ledger_shaped(node):
    """The literal ``{str: int}`` a value node holds, or None."""
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    if not isinstance(value, dict) or not value:
        return None
    if not all(isinstance(k, str) for k in value):
        return None
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in value.values()):
        return None
    return value


def _vanished_ledgers(root, test_file, name, keys):
    """``{"<test file>:<name>": value}`` for every ledger HEAD holds in a
    Python file under the test file's directory that shares at least one
    path with ``keys`` and is gone from the working tree: its file gone, or
    its name no longer assigned there. These are the predecessors a ledger
    born under a new name or in a new file must answer to."""
    if not keys:
        return {}
    root = Path(root)
    rel = Path(test_file).resolve().relative_to(root.resolve()).as_posix()
    directory = PurePosixPath(rel).parent.as_posix()
    patterns = []
    for key in sorted(keys):
        patterns += ["-e", key]
    raw = _git(
        root, "grep", "-l", "-z", "-F", *patterns, "HEAD", "--", directory,
        ok=(0, 1),
    )
    vanished = {}
    for entry in _split_z(raw):
        path = entry.split(":", 1)[1]
        if not path.endswith(".py"):
            continue
        try:
            at_head = _module_values(_git(root, "show", f"HEAD:{path}").decode("utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        here = None
        for held, node in at_head.items():
            value = _ledger_shaped(node)
            if value is None or not set(value) & set(keys):
                continue
            if here is None:
                here = {}
                if (root / path).is_file():
                    try:
                        here = _module_values(read(path, root=root))
                    except (SyntaxError, UnicodeDecodeError):
                        here = {}
            if held in here:
                continue
            vanished[f"{path}:{held}"] = value
    return vanished


def _renames_since_head(root, within):
    """``{new path: old path}`` for every rename under ``within`` that
    ``git diff -M HEAD`` shows once the listing's untracked files are visible
    to it.

    Plain ``git diff HEAD`` sees only the paths the index holds, so a move
    made without ``git mv`` shows as a deletion alone. The listing counts
    untracked files, so the rename reader sees them too: they are marked
    intent-to-add in a scratch copy of the index under ``$TMPDIR``, never in
    the index itself, and git's own rename detection pairs them.
    """
    untracked = _split_z(_git(
        root, "ls-files", "-z", "--others", "--exclude-standard", "--", within,
    ))
    diff = (
        "diff", "-M", "--name-status", "-z", "--no-color", "--no-ext-diff",
        "HEAD", "--", within,
    )
    if untracked:
        real = Path(_git(
            root, "rev-parse", "--path-format=absolute", "--git-path", "index",
        ).decode("utf-8").strip())
        with tempfile.TemporaryDirectory(prefix="oo_ratchet_index_") as tmp:
            scratch = Path(tmp) / "index"
            shutil.copyfile(real, scratch)
            _git(
                root, "add", "--intent-to-add", "--pathspec-from-file=-",
                "--pathspec-file-nul", index=scratch,
                stdin=b"\0".join(path.encode("utf-8") for path in untracked),
            )
            raw = _git(root, *diff, index=scratch)
    else:
        raw = _git(root, *diff)
    fields = _split_z(raw)
    renames = {}
    index = 0
    while index < len(fields):
        status = fields[index]
        if status[:1] in ("R", "C"):
            old, new = fields[index + 1], fields[index + 2]
            if status[:1] == "R":
                renames[new] = old
            index += 3
        else:
            index += 2
    return renames


def check_ledger(name, ledger, count_fn, fixture, *, test_file,
                 suffixes=(".svelte",), within=SOURCE, exclude=(), root=REPO):
    """Hold one census to its ledger; raise with every finding, or return it.

    ``name`` is the module-level name the ledger is assigned to in
    ``test_file`` (the caller's ``__file__``), which is how its value at HEAD
    is found. ``count_fn(path, text)`` returns a count. ``fixture`` is the
    positive sample, a text (counted as a file under ``within`` with the
    first suffix) or a ``(path, text)`` pair.
    """
    root = Path(root)
    listed = files(suffixes, within=within, exclude=exclude, root=root)

    if isinstance(fixture, tuple):
        fixture_path, fixture_text = fixture
    else:
        fixture_path, fixture_text = f"{within}/fixture{tuple(suffixes)[0]}", fixture
    fixture_count = count_fn(fixture_path, fixture_text)

    problems = []
    if fixture_count < 1:
        problems.append(
            f"the count function reads {fixture_count} on its positive "
            f"fixture: the probe is blind, and its zeros mean nothing"
        )

    counts = {}
    for path in listed:
        found = count_fn(path, read(path, root=root))
        if found:
            counts[path] = found

    for path, found in counts.items():
        entry = ledger.get(path)
        if entry is None:
            problems.append(
                f"{path} counts {found} and is not in the ledger "
                f"(a file not in the ledger counts 0)"
            )
        elif found > entry:
            problems.append(f"{path} counts {found}, above its entry {entry}")

    for path, entry in ledger.items():
        found = counts.get(path, 0)
        if entry > found:
            problems.append(
                f"{path} entry {entry} is above its count {found}: lower the ledger"
            )

    at_head = _ledger_at_head(root, test_file, name)
    compared_with = ""
    if at_head is not None:
        compared_with = (
            Path(test_file).resolve().relative_to(root.resolve()).as_posix()
            + f":{name}"
        )
    else:
        vanished = _vanished_ledgers(root, test_file, name, set(ledger))
        if len(vanished) == 1:
            ((compared_with, at_head),) = vanished.items()
        elif vanished:
            problems.append(
                f"{name} is not at HEAD, and several ledgers HEAD holds share "
                f"its files and are gone: {', '.join(sorted(vanished))}; a "
                f"ledger is born once, and never re-born under a new name"
            )
    if at_head is None:
        head = "born"
    else:
        head = "compared"
        renames = None
        for path, entry in ledger.items():
            if path in at_head:
                if entry > at_head[path]:
                    problems.append(
                        f"{path} entry {entry} is above its value at HEAD "
                        f"{at_head[path]} ({compared_with}): a ledger never rises"
                    )
                continue
            if renames is None:
                renames = _renames_since_head(root, within)
            old = renames.get(path)
            if old is None:
                problems.append(
                    f"{path} entry {entry} is new since HEAD and is not a rename "
                    f"git diff -M HEAD shows (a moved file carries its entry "
                    f"only when git's similarity pairs it with the file it was)"
                )
            elif old not in at_head:
                problems.append(
                    f"{path} entry {entry} is new since HEAD: it was renamed "
                    f"from {old}, which had no entry at HEAD"
                )
            elif entry > at_head[old]:
                problems.append(
                    f"{path} entry {entry}, renamed from {old}, is above that "
                    f"file's entry at HEAD {at_head[old]}: a rename carries "
                    f"its entry, never a higher one"
                )

    if problems:
        raise AssertionError(
            f"ratchet {name} ({len(listed)} files listed):\n  "
            + "\n  ".join(problems)
        )
    return Census(name, tuple(listed), counts, head, fixture_count, compared_with)


def node_version():
    """The installed Node as ``(major, minor)``; raises when there is none."""
    exe = shutil.which("node")
    if exe is None:
        raise RuntimeError(
            "node is required for the frontend contracts "
            "(>= 22.6 with TypeScript type stripping)"
        )
    out = subprocess.run(
        [exe, "--version"], capture_output=True, text=True, check=True,
    ).stdout.strip().lstrip("v")
    major, minor = out.split(".")[:2]
    return int(major), int(minor)


def run_ts(modules, driver, clause, *, env=None, timeout=60):
    """Run one clause of a driver against dependency-free TypeScript modules.

    ``modules`` maps an environment variable to a module path (absolute, or
    relative to the repository); the driver, an ES module, imports each one
    from ``process.env[VAR]`` (a file URL). The clause is the driver's first
    argument, and it passes only by printing the line ``PASS <clause>`` and
    exiting 0. Returns the driver's standard output.

    Its refusals are proven by MK18 (tests/test_markdown_render_contracts.py):
    a driver that exits 0 without its ``PASS`` line, one that prints it and
    exits non-zero, one that passes another clause, and an absent module.
    """
    version = node_version()
    if version < NODE_MIN:
        raise RuntimeError(
            f"node >= {NODE_MIN[0]}.{NODE_MIN[1]} required for type stripping "
            f"(found {version[0]}.{version[1]})"
        )
    run_env = dict(os.environ)
    run_env.update(env or {})
    for var, module in modules.items():
        path = Path(module)
        if not path.is_absolute():
            path = REPO / path
        if not path.is_file():
            raise AssertionError(f"module under contract is absent: {path}")
        run_env[var] = path.resolve().as_uri()
    with tempfile.TemporaryDirectory(prefix="oo_run_ts_") as tmp:
        script = Path(tmp) / "driver.mjs"
        script.write_text(driver, encoding="utf-8")
        proc = None
        for extra in ([], ["--experimental-strip-types"]):
            proc = subprocess.run(
                ["node", *extra, str(script), clause],
                capture_output=True, text=True, env=run_env, timeout=timeout,
            )
            if "ERR_UNKNOWN_FILE_EXTENSION" not in (proc.stderr or ""):
                break
    if proc.returncode != 0 or f"PASS {clause}" not in proc.stdout.splitlines():
        raise AssertionError(
            f"clause {clause} failed (rc={proc.returncode}):\n"
            f"{proc.stdout}{proc.stderr}"
        )
    return proc.stdout


STUBS = Path(__file__).resolve().parent / "_frontend_stubs"
_STUB_NAMES = ("navigation", "stores", "environment")
_SSR_DIR = "frontend/.oo_ssr"
_SSR_TAG = "OO_SSR "

# The server-rendering process: vite in middleware mode (no port, no file
# watcher, no dependency optimizer, no config file, no PostCSS), the Svelte
# plugin with TypeScript preprocessing, ``$lib`` and the three ``$app``
# modules aliased. It reads one JSON request per line and answers each on a
# tagged line, an error included, so a failed render is never an empty one.
_SSR_SERVER = r"""
import path from 'node:path';
import readline from 'node:readline';
import { createServer } from 'vite';
import { svelte, vitePreprocess } from '@sveltejs/vite-plugin-svelte';

const [root, cacheDir] = process.argv.slice(2);
const stubs = path.join(root, '.oo_ssr', 'app');
const say = (payload) => process.stdout.write('OO_SSR ' + JSON.stringify(payload) + '\n');
const describe = (error) => String((error && error.stack) || error);

let server;
try {
    server = await createServer({
        root,
        cacheDir,
        configFile: false,
        logLevel: 'silent',
        clearScreen: false,
        appType: 'custom',
        server: { middlewareMode: true, hmr: false, watch: null },
        optimizeDeps: { noDiscovery: true, include: [] },
        css: { postcss: {} },
        resolve: {
            alias: [
                { find: /^\$lib(?=\/|$)/, replacement: path.join(root, 'src', 'lib') },
                { find: '$app/navigation', replacement: path.join(stubs, 'navigation.js') },
                { find: '$app/stores', replacement: path.join(stubs, 'stores.js') },
                { find: '$app/environment', replacement: path.join(stubs, 'environment.js') },
            ],
        },
        plugins: [svelte({ configFile: false, preprocess: vitePreprocess({ style: false }) })],
    });
} catch (error) {
    say({ id: 0, error: describe(error) });
    process.exit(1);
}
say({ id: 0, ready: true });

const lines = readline.createInterface({ input: process.stdin });
for await (const line of lines) {
    let request;
    try {
        request = JSON.parse(line);
    } catch {
        continue;
    }
    try {
        const module = await server.ssrLoadModule(path.join(root, request.path));
        const component = module.default;
        if (!component || typeof component.render !== 'function') {
            throw new Error(request.path + ' exports no component that renders on the server');
        }
        const out = component.render(request.props || {});
        say({ id: request.id, html: out.html, head: out.head, css: (out.css && out.css.code) || '' });
    } catch (error) {
        say({ id: request.id, error: describe(error) });
    }
}
await server.close();
"""


@dataclass(frozen=True)
class Rendered:
    """What a component compiled for the server emitted."""

    html: str
    head: str
    css: str


class SsrServer:
    """One vite server rendering components for the server, on a copy.

    The copy is a ``frontend_copy`` under ``$TMPDIR`` with the three ``$app``
    stubs of ``tests/_frontend_stubs/`` and the server script planted under
    ``.oo_ssr/``; vite's root and cache are both in it, so nothing is written
    in the tree or beside the installed packages. Components are named by
    their repository path (``frontend/src/...``).
    """

    def __init__(self, root=REPO, *, timeout=60):
        self._scratch = tempfile.TemporaryDirectory(prefix="oo_ssr_")
        planted = {
            f"{_SSR_DIR}/app/{name}.js": (STUBS / f"{name}.js").read_text(encoding="utf-8")
            for name in _STUB_NAMES
        }
        planted[f"{_SSR_DIR}/server.mjs"] = _SSR_SERVER
        self.root = frontend_copy(self._scratch.name, root=root, extra=planted)
        # The tsconfig extends the one svelte-kit generates; the TypeScript
        # preprocessing reads it, as the app's build does.
        sync = subprocess.run(
            [str(self.root / "node_modules" / ".bin" / "svelte-kit"), "sync"],
            cwd=self.root, capture_output=True, text=True, timeout=timeout,
        )
        if sync.returncode != 0:
            self._scratch.cleanup()
            raise RuntimeError(
                f"svelte-kit sync failed (rc={sync.returncode}):\n{sync.stdout}{sync.stderr}"
            )
        self._lines = queue.Queue()
        self._errors = []
        self._next = 0
        self._proc = subprocess.Popen(
            ["node", str(self.root / ".oo_ssr" / "server.mjs"), str(self.root),
             str(Path(self._scratch.name) / "vite-cache")],
            cwd=self.root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
        )
        threading.Thread(
            target=self._pump, args=(self._proc.stdout, self._lines), daemon=True,
        ).start()
        threading.Thread(
            target=self._pump, args=(self._proc.stderr, self._errors), daemon=True,
        ).start()
        ready = self._answer(0, timeout)
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(f"the server-rendering process did not start:\n{ready}")

    @staticmethod
    def _pump(stream, sink):
        for line in stream:
            if isinstance(sink, list):
                sink.append(line)
            elif line.startswith(_SSR_TAG):
                sink.put(json.loads(line[len(_SSR_TAG):]))

    def _answer(self, ident, timeout):
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise RuntimeError(
                    f"no answer from the server-rendering process within {timeout}s:\n"
                    + "".join(self._errors[-40:])
                )
            try:
                answer = self._lines.get(timeout=min(left, 0.5))
            except queue.Empty:
                if self._proc.poll() is not None and self._lines.empty():
                    raise RuntimeError(
                        f"the server-rendering process exited ({self._proc.returncode}):\n"
                        + "".join(self._errors[-40:])
                    ) from None
                continue
            if answer.get("id") == ident:
                return answer

    def plant(self, path, text):
        """Writes ``text`` at the repository path ``path`` in the copy only.

        A path the copy already holds is refused, so a plant never hides a
        file of the tree, and a plant is never rewritten under a module
        vite has already loaded."""
        parts = PurePosixPath(path).parts
        if len(parts) < 2 or parts[0] != "frontend" or ".." in parts or parts[1] == "node_modules":
            raise AssertionError(f"cannot plant {path}: not a frontend path")
        target = self.root.joinpath(*parts[1:])
        if target.exists() or target.is_symlink():
            raise AssertionError(f"cannot plant {path}: the copy already holds it")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target

    def render(self, path, props=None, *, timeout=30):
        """Renders the component at the repository path ``path`` with
        ``props``. A render that fails raises, never reads as empty."""
        parts = PurePosixPath(path).parts
        if len(parts) < 2 or parts[0] != "frontend" or ".." in parts:
            raise AssertionError(f"cannot render {path}: not a frontend path")
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"the server-rendering process has exited ({self._proc.returncode}):\n"
                + "".join(self._errors[-40:])
            )
        self._next += 1
        ident = self._next
        request = {"id": ident, "path": str(PurePosixPath(*parts[1:])), "props": props or {}}
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        answer = self._answer(ident, timeout)
        if "error" in answer:
            raise AssertionError(f"{path} did not render on the server:\n{answer['error']}")
        return Rendered(answer["html"], answer["head"], answer["css"])

    def close(self):
        proc = self._proc
        if proc.poll() is None:
            try:
                proc.stdin.close()
                proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                proc.kill()
                proc.wait(timeout=10)
        self._scratch.cleanup()


_SSR = None


def ssr():
    """The session's server renderer, started on first use and closed at exit.

    One vite server per test session (``SsrServer``): the Svelte plugin,
    ``$lib`` aliased, ``$app/navigation``, ``$app/stores`` and
    ``$app/environment`` aliased to the stubs under ``tests/_frontend_stubs/``,
    vite's root and cache in a copy under ``$TMPDIR``, nothing written in the
    tree. ``plant(path, text)`` adds a file to the copy; ``render(path,
    props)`` returns what the component emits (``html``, ``head``, ``css``).
    The app is client-rendered only, so what it proves is what the template
    emits when compiled for the server, not what the browser build does.
    Needs Node and ``frontend/node_modules``; without them it raises.
    """
    global _SSR
    if _SSR is None:
        node_version()
        _SSR = SsrServer()
        atexit.register(_SSR.close)
    return _SSR


_COMPLETED = re.compile(
    r"^\d+ COMPLETED (\d+) FILES (\d+) ERRORS (\d+) WARNINGS (\d+) FILES_WITH_PROBLEMS\s*$"
)
_ERROR = re.compile(r'^\d+ ERROR ("(?:[^"\\]|\\.)*") ')


def frontend_copy(dest, *, root=REPO, extra=None):
    """A working copy of the frontend at ``dest/frontend``, which it returns.

    The files copied are the frontend's listed files (the census listing
    rule): a file being written is copied, an ignored one never is.
    ``node_modules`` is a real directory holding one link per installed
    package (``.bin`` included, a tool's dot-named cache not), so whatever a
    tool writes beside the dependencies lands in the copy and never in the
    installed tree. ``extra`` maps repository paths under ``frontend/`` to
    texts planted in the copy only; a path the copy already holds is
    refused, so a plant never hides a real file. Without
    ``frontend/node_modules`` it raises, owed.
    """
    root = Path(root)
    modules = root / "frontend" / "node_modules"
    if not modules.is_dir():
        raise RuntimeError("OWED: frontend/node_modules absent")
    listing = _split_z(_git(
        root, "ls-files", "-z", "--cached", "--others", "--exclude-standard",
        "--", "frontend",
    ))
    copy = Path(dest) / "frontend"
    for rel in sorted(set(listing)):
        if rel == "frontend/node_modules" or rel.startswith("frontend/node_modules/"):
            continue
        source = root / rel
        if not source.is_file():
            continue
        target = copy / Path(rel).relative_to("frontend")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    linked = copy / "node_modules"
    linked.mkdir(parents=True)
    installed = modules.resolve()
    for entry in sorted(os.listdir(installed)):
        if entry.startswith(".") and entry != ".bin":
            continue
        (linked / entry).symlink_to(installed / entry)
    for rel, text in (extra or {}).items():
        parts = Path(rel).parts
        if (
            len(parts) < 2 or parts[0] != "frontend" or parts[1] == "node_modules"
            or ".." in parts
        ):
            raise AssertionError(f"cannot plant {rel}: not a frontend path")
        target = copy / Path(*parts[1:])
        if target.exists() or target.is_symlink():
            raise AssertionError(f"cannot plant {rel}: the copy already holds it")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return copy


class CheckCounts(dict):
    """svelte-check errors per file, and ``checked``: the number of files
    its ``COMPLETED`` line says it checked."""

    checked = 0


def svelte_check_counts(root=REPO, *, extra=None, timeout=600):
    """svelte-check errors per file, as ``{repository path: count}``.

    Runs on a ``frontend_copy`` under ``$TMPDIR``: ``svelte-kit sync`` then
    ``svelte-check --tsconfig ./tsconfig.json --output machine`` from the
    local binaries (npm's cache is never touched), and reads the ``ERROR``
    lines. The ``COMPLETED`` line, which has no plural form, must be there
    and its error total must equal the lines read; otherwise the run is
    refused, never read as zero. Its file total is kept as ``checked``.
    ``extra`` plants files in the copy (see ``frontend_copy``); their errors
    are counted under their own paths.
    """
    with tempfile.TemporaryDirectory(prefix="oo_svelte_check_") as tmp:
        copy = frontend_copy(tmp, root=root, extra=extra)
        tools = copy / "node_modules" / ".bin"
        sync = subprocess.run(
            [str(tools / "svelte-kit"), "sync"],
            cwd=copy, capture_output=True, text=True, timeout=timeout,
        )
        if sync.returncode != 0:
            raise RuntimeError(
                f"svelte-kit sync failed (rc={sync.returncode}):\n"
                f"{sync.stdout}{sync.stderr}"
            )
        proc = subprocess.run(
            [str(tools / "svelte-check"), "--tsconfig", "./tsconfig.json",
             "--output", "machine"],
            cwd=copy, capture_output=True, text=True, timeout=timeout,
        )
    counts = CheckCounts()
    completed = None
    for line in proc.stdout.splitlines():
        match = _ERROR.match(line)
        if match:
            name = json.loads(match.group(1))
            if name.startswith("./"):
                name = name[2:]
            path = "frontend/" + name
            counts[path] = counts.get(path, 0) + 1
            continue
        match = _COMPLETED.match(line)
        if match:
            counts.checked = int(match.group(1))
            completed = int(match.group(2))
    if completed is None:
        raise AssertionError(
            f"svelte-check printed no COMPLETED line (rc={proc.returncode}); "
            f"its count cannot be read:\n{proc.stdout[-2000:]}{proc.stderr[-2000:]}"
        )
    read_total = sum(counts.values())
    if read_total != completed:
        raise AssertionError(
            f"svelte-check reports {completed} errors and {read_total} ERROR "
            f"lines were read: the parser missed some"
        )
    return counts
